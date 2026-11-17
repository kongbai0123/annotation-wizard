"""Read-only video approval dialog and cancellable, staged detection job."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
VIDEO_TYPES = [("影片", "*.mp4 *.mov *.avi *.m4v *.mkv *.wmv *.webm *.mts *.m2ts"),
               ("所有檔案", "*.*")]
VIDEO_SUFFIXES = {suffix.removeprefix("*") for suffix in VIDEO_TYPES[0][1].split()}
MAX_SAMPLED_FRAMES = 20_000
FONT = "Microsoft JhengHei UI"
BACKGROUND, PANEL, TEXT, MUTED = "#101820", "#18242c", "#ebf2f3", "#a8bbc3"


@dataclass(frozen=True)
class FileStamp:
    size: int
    modified_ns: int
    sha256: str


@dataclass(frozen=True)
class VideoPreview:
    path: Path
    stamp: FileStamp
    width: int
    height: int
    fps: float
    frames: int
    stills: tuple[tuple[float, Image.Image], ...]


@dataclass(frozen=True)
class VideoApproval:
    video: Path
    video_sha256: str
    model: Path
    model_sha256: str
    interpreter: Path
    sample_fps: float
    confidence: float


def _local_file(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if str(resolved).startswith(("\\\\", "//")) or not resolved.is_file():
        raise ValueError("請選擇本機磁碟上的檔案。")
    return resolved


def _fingerprint(path: Path, stop: Event) -> FileStamp:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            if stop.is_set():
                raise InterruptedError("已停止預覽。")
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("檔案在審查期間有變更，請重新選取。")
    return FileStamp(after.st_size, after.st_mtime_ns, digest.hexdigest())


def _still(frame) -> Image.Image:
    height, width = frame.shape[:2]
    factor = min(1, 640 / width, 270 / height)
    if factor < 1:
        frame = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                           interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def inspect_video(path: Path, stop: Event) -> VideoPreview:
    """Read selected source bytes, but do not create or copy any files."""
    path = _local_file(path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("請選擇 MP4、MOV、AVI 或其他支援的影片檔。")
    before = path.stat()
    capture = cv2.VideoCapture(os.fspath(path))
    try:
        ok, first = capture.read()
        if not ok or first is None:
            raise ValueError("影片無法讀取。請確認檔案能播放，再選擇影片。")
        height, width = first.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = int(count) if math.isfinite(count) and count >= 1 else 0
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("影片沒有可用的畫格率，請先轉成能讀取 FPS 的影片。")
        stills = [(0.0, _still(first))]
        for index in sorted({round((frames - 1) * part) for part in (0.25, 0.5, 0.75, 0.95)} - {0}):
            if stop.is_set() or index < 0:
                break
            if capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                ok, frame = capture.read()
                if ok and frame is not None:
                    stills.append((index / fps, _still(frame)))
    finally:
        capture.release()
    stamp = _fingerprint(path, stop)
    if (stamp.size, stamp.modified_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError("影片在預覽期間有變更，請重新選取。")
    return VideoPreview(path, stamp, width, height, fps, frames, tuple(stills))


def _inspect_model(path: Path, stop: Event) -> tuple[Path, FileStamp]:
    path = _local_file(path)
    if path.suffix.lower() != ".pt":
        raise ValueError("請選擇副檔名為 .pt 的本機 YOLO 偵測模型。")
    stamp = _fingerprint(path, stop)
    if not stamp.size:
        raise ValueError("模型檔是空的，請重新選擇。")
    return path, stamp


def resolve_runtime() -> Path | None:
    override = os.environ.get("ANNOTATION_WIZARD_VIDEO_PYTHON")
    if override:
        configured = Path(override)
        return configured.resolve() if configured.is_file() else None
    own = Path(sys.executable).with_name("python.exe")
    if own.is_file() and (own.parent.parent / "Lib" / "site-packages" / "ultralytics").is_dir():
        return own.resolve()
    shared = ROOT.parent / "vision-workbench" / ".venv-models" / "ultralytics" / "Scripts" / "python.exe"
    return shared.resolve() if shared.is_file() else None


def _clock(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


class VideoReviewDialog:
    def __init__(self, parent, path: Path):
        self.window = tk.Toplevel(parent)
        self.window.title("審查影片")
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        self.window.configure(bg=BACKGROUND)
        width = min(742, max(560, self.window.winfo_screenwidth() - 40))
        height = min(620, max(490, self.window.winfo_screenheight() - 100))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(560, 490)
        self.stop = Event()
        self.events = SimpleQueue()
        self.runtime = resolve_runtime()
        self.preview: VideoPreview | None = None
        self.model: tuple[Path, FileStamp] | None = None
        self.model_request = 0
        self.index = 0
        self.approved: VideoApproval | None = None
        self.picture = None
        self.info = tk.StringVar(value="正在讀取影片和核對檔案內容…")
        self.thumb_text = tk.StringVar(value="等待預覽")
        self.model_text = tk.StringVar(value="尚未選擇模型")
        self.sample = tk.StringVar(value="2")
        self.confidence = tk.StringVar(value="0.35")
        self.status = tk.StringVar(value="請先確認影片內容，再選擇對應類別的本機模型。")

        footer = tk.Frame(self.window, bg=BACKGROUND)
        footer.pack(side="bottom", fill="x", padx=17, pady=(4, 14))
        self._label(footer, variable=self.status, color=MUTED, wrap=700).pack(fill="x", pady=(0, 8))
        buttons = tk.Frame(footer, bg=BACKGROUND)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="取消", command=self.close).pack(side="right", padx=(8, 0))
        self.confirm_button = ttk.Button(buttons, text="確認影片並匯入", style="Primary.TButton",
                                         command=self.confirm, state="disabled")
        self.confirm_button.pack(side="right")

        heading = tk.Frame(self.window, bg=BACKGROUND)
        heading.pack(fill="x", padx=17, pady=(13, 5))
        self._label(heading, "確認這一部影片", size=16, bold=True).pack(fill="x")
        self._label(heading, "此畫面僅預覽。按下確認後才會擷取畫格、偵測和匯入。", color=MUTED).pack(fill="x", pady=(3, 6))
        video_text = tk.StringVar(value=str(path))
        ttk.Entry(heading, textvariable=video_text, state="readonly").pack(fill="x")
        self._label(heading, variable=self.info, color=MUTED, wrap=700).pack(fill="x", pady=(4, 0))

        player = tk.Frame(self.window, bg=PANEL)
        player.pack(fill="both", expand=True, padx=17, pady=5)
        navigation = tk.Frame(player, bg=PANEL)
        navigation.pack(side="bottom", fill="x", padx=8, pady=5)
        self.previous = ttk.Button(navigation, text="上一張預覽", command=lambda: self._show(-1), state="disabled")
        self.previous.pack(side="left")
        self.following = ttk.Button(navigation, text="下一張預覽", command=lambda: self._show(1), state="disabled")
        self.following.pack(side="left", padx=5)
        self._label(navigation, variable=self.thumb_text, color=MUTED, bg=PANEL).pack(side="left", padx=4)
        self.play = ttk.Button(navigation, text="播放原片", command=self._play, state="disabled")
        self.play.pack(side="right")
        self.display = tk.Label(player, bg=PANEL, fg=MUTED, text="正在載入影片預覽", font=(FONT, 10))
        self.display.pack(fill="both", expand=True)
        self.display.bind("<Configure>", self._render)

        settings = tk.Frame(self.window, bg=BACKGROUND)
        settings.pack(fill="x", padx=17, pady=4)
        ttk.Button(settings, text="選擇本機 YOLO .pt", command=self._choose_model).pack(side="left", padx=(0, 8))
        ttk.Entry(settings, textvariable=self.model_text, state="readonly").pack(side="left", expand=True, fill="x")
        numbers = tk.Frame(self.window, bg=BACKGROUND)
        numbers.pack(fill="x", padx=17, pady=(0, 4))
        self._label(numbers, "儲存畫格／秒").pack(side="left")
        ttk.Entry(numbers, textvariable=self.sample, width=7).pack(side="left", padx=(6, 18))
        self._label(numbers, "偵測信心門檻").pack(side="left")
        ttk.Entry(numbers, textvariable=self.confidence, width=7).pack(side="left", padx=6)

        self.sample.trace_add("write", self._update_confirm)
        self.confidence.trace_add("write", self._update_confirm)
        Thread(target=self._read, args=("video", 0, lambda: inspect_video(path, self.stop)), daemon=True).start()
        self.poll_id = self.window.after(60, self._collect)

    @staticmethod
    def _label(parent, text="", *, variable=None, color=TEXT, bg=BACKGROUND, size=10, bold=False, wrap=0):
        return tk.Label(parent, text=text, textvariable=variable, fg=color, bg=bg, wraplength=wrap,
                        anchor="w", justify="left", font=(FONT, size, "bold" if bold else "normal"))

    def show(self) -> VideoApproval | None:
        self.window.grab_set()
        self.window.wait_window()
        return self.approved

    def close(self):
        self.stop.set()
        self.window.after_cancel(self.poll_id)
        self.window.destroy()

    def _read(self, role, request, read):
        try:
            self.events.put((role, request, read(), None))
        except Exception as exc:
            self.events.put((role, request, None, str(exc)))

    def _collect(self):
        while True:
            try:
                role, request, value, error = self.events.get_nowait()
            except Empty:
                break
            if role == "video":
                if error:
                    self.info.set(error)
                    self.display.configure(text="無法讀取預覽")
                else:
                    self.preview = value
                    duration = _clock(value.frames / value.fps) if value.frames else "未知"
                    self.info.set(f"{value.width} × {value.height}　{value.fps:.3g} FPS　長度 {duration}　"
                                  f"{value.stamp.size / 1024**2:,.1f} MB　SHA-256 {value.stamp.sha256[:12]}…")
                    self.sample.set(f"{min(2, value.fps):.6g}")
                    self.play.state(["!disabled"] if hasattr(os, "startfile") else ["disabled"])
                    self._show(0)
            elif request == self.model_request:
                if error:
                    self.model_text.set(error)
                else:
                    self.model = value
                    self.model_text.set(str(value[0]))
            self._update_confirm()
        self.poll_id = self.window.after(60, self._collect)

    def _show(self, delta):
        if self.preview is None:
            return
        self.index = max(0, min(len(self.preview.stills) - 1, self.index + delta))
        self.previous.state(["!disabled"] if self.index > 0 else ["disabled"])
        self.following.state(["!disabled"] if self.index < len(self.preview.stills)-1 else ["disabled"])
        self.thumb_text.set(f"約 {_clock(self.preview.stills[self.index][0])}　"
                            f"{self.index + 1}/{len(self.preview.stills)}")
        self._render()

    def _render(self, _event=None):
        if self.preview is None:
            return
        frame = self.preview.stills[self.index][1].copy()
        frame.thumbnail((max(20, self.display.winfo_width() - 10), max(20, self.display.winfo_height() - 8)),
                        Image.Resampling.LANCZOS)
        self.picture = ImageTk.PhotoImage(frame, master=self.window)
        self.display.configure(image=self.picture, text="")

    def _play(self):
        if self.preview is None:
            return
        try:
            os.startfile(self.preview.path)
        except OSError as exc:
            self.status.set(f"無法播放原片。{exc}")

    def _choose_model(self):
        chosen = filedialog.askopenfilename(parent=self.window, title="選擇信任的本機 YOLO 偵測模型",
                                            filetypes=[("YOLO 權重", "*.pt")])
        if not chosen:
            return
        self.model, self.model_request = None, self.model_request + 1
        self.model_text.set(f"正在核對 {chosen}…")
        self._update_confirm()
        Thread(target=self._read, args=("model", self.model_request, lambda: _inspect_model(Path(chosen), self.stop)),
               daemon=True).start()

    def _parameters(self) -> tuple[float, float]:
        try:
            sample, confidence = float(self.sample.get()), float(self.confidence.get())
        except ValueError as exc:
            raise ValueError("請輸入有效的畫格率與信心門檻。") from exc
        if not 0.1 <= sample <= min(self.preview.fps + 1e-6, 60):
            raise ValueError("儲存畫格率須在 0.1 到影片原始 FPS 之間，上限 60。")
        if not 0.01 <= confidence <= 1:
            raise ValueError("信心門檻須在 0.01 到 1 之間。")
        if self.preview.frames and math.ceil(self.preview.frames * sample / self.preview.fps) > MAX_SAMPLED_FRAMES:
            raise ValueError("預計擷取超過 20,000 張畫格，請降低儲存畫格率。")
        return min(sample, self.preview.fps), confidence

    def _update_confirm(self, *_args):
        ready = False
        if self.runtime is None:
            self.status.set("找不到本機 Ultralytics 執行環境。請設定 ANNOTATION_WIZARD_VIDEO_PYTHON。")
        elif self.preview is None:
            self.status.set("影片預覽與檔案核對完成後才能確認。")
        elif self.model is None:
            self.status.set("請選擇要使用且可信任的 YOLO 偵測模型。")
        else:
            try:
                sample, _ = self._parameters()
                preview = self.preview
                output = (f"預計儲存約 {math.ceil(preview.frames * sample / preview.fps):,} 張畫格。"
                          if preview.frames else "取樣張數未知。")
                self.status.set(output + "每張匯入畫格須複核後才可匯出。")
                ready = True
            except ValueError as exc:
                self.status.set(str(exc))
        self.confirm_button.state(["!disabled"] if ready else ["disabled"])

    def confirm(self):
        if self.preview is None or self.model is None or self.runtime is None or self.confirm_button.instate(["disabled"]):
            return
        try:
            sample, confidence = self._parameters()
            for path, stamp in ((self.preview.path, self.preview.stamp), self.model):
                now = path.stat()
                if (now.st_size, now.st_mtime_ns) != (stamp.size, stamp.modified_ns):
                    raise ValueError("影片或模型檔案在審查後有變更，請重新選擇。")
        except (OSError, ValueError) as exc:
            self.status.set(str(exc))
            self.confirm_button.state(["disabled"])
            return
        self.approved = VideoApproval(self.preview.path, self.preview.stamp.sha256, self.model[0],
                                      self.model[1].sha256, self.runtime, sample, confidence)
        self.close()


class VideoImport:
    """Own the subprocess, hold staged files until completion and commit once."""

    def __init__(self, approval: VideoApproval):
        self.approval = approval
        self.stop = Event()
        self._commit_lock = Lock()
        self.phase = "working"
        self.events = SimpleQueue()

    def cancel(self) -> bool:
        with self._commit_lock:
            if self.phase != "working":
                return False
            self.stop.set()
            return True

    def _progress(self, stream, tail):
        for line in stream:
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    if event.get("event") == "error" and event.get("message"):
                        tail.append(str(event["message"]))
                    self.events.put(event)
            except (ValueError, TypeError):
                continue

    @staticmethod
    def _logs(stream, tail):
        for line in stream:
            if line.strip():
                tail.append(line.strip())

    @staticmethod
    def _terminate(process):
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
            try:
                # Stop the launched venv shim and its worker child together.
                subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
                if process.poll() is not None:
                    return
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except ProcessLookupError:
            pass

    def run(self, project) -> dict | None:
        if self.stop.is_set():
            return None
        with TemporaryDirectory(prefix="annotation-video-") as temp:
            directory = Path(temp)
            out = directory / "result"
            approval = self.approval
            cmd = [str(approval.interpreter), str(ROOT / "video_worker.py"), "--video", str(approval.video),
                   "--model", str(approval.model), "--out", str(out), "--sample-fps", str(approval.sample_fps),
                   "--conf", str(approval.confidence), "--approved-video-sha256", approval.video_sha256,
                   "--approved-model-sha256", approval.model_sha256]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            env["YOLO_CONFIG_DIR"] = str(directory / "worker-settings")
            tail = deque(maxlen=8)
            with subprocess.Popen(cmd, cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                stdout = Thread(target=self._progress, args=(process.stdout, tail), daemon=True)
                stderr = Thread(target=self._logs, args=(process.stderr, tail), daemon=True)
                stdout.start()
                stderr.start()
                try:
                    while process.poll() is None and not self.stop.wait(0.08):
                        continue
                finally:
                    if process.poll() is None:
                        self._terminate(process)
                    stdout.join(timeout=2)
                    stderr.join(timeout=2)
                if self.stop.is_set():
                    return None
                if process.returncode or not (out / "result.json").is_file():
                    details = "\n".join(tail)
                    raise RuntimeError("影片標註未完成。" + (f"\n{details[-1600:]}" if details else ""))
            with self._commit_lock:
                if self.stop.is_set():
                    return None
                self.phase = "commit"
            self.events.put({"event": "commit"})
            return project.import_video_annotations(out / "result.json",
                                                    approved_video_sha256=approval.video_sha256,
                                                    approved_model_sha256=approval.model_sha256)
