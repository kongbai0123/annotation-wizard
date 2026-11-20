"""Bridge saved jobs to the offline Windows workers without exposing project truth."""

from hashlib import file_digest
import json
import os
from pathlib import Path
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
ACTIVE_PROFILES = {"infer": "AnnotationWizardGenericInferenceV1", "train": "AnnotationWizardGenericTrainingV1"}
INFERENCE_PHASES = {"scene_scan": "Qwen 掃描原圖", "dino": "MM 找候選框", "qwen": "Qwen 逐框裁決",
                    "dino_retry": "MM 局部重找", "qwen_retry": "Qwen 重查", "sam_qc": "SAM 框界抽查"}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def stage_inference(job, target):
    """Only the sealed manifest and its listed inference images leave the host project."""
    (target / "images").mkdir()
    copyfile(job.manifest_path, target / "manifest.json")
    for frame in job.manifest["frames"]:
        name = Path(frame["image"])
        if name != Path("images") / f"{frame['id']}.png":
            raise ValueError("作業只接受專案內已核對的標準畫格路徑。")
        source = (job.project_dir / name).resolve(strict=True)
        if not source.is_relative_to(job.project_dir / "images"):
            raise ValueError("畫格指向專案外部。")
        staged = target / name
        copyfile(source, staged)
        if digest(staged) != frame["image_sha256"]:
            raise ValueError(f"畫格在執行前已變更：{frame['name']}")


def _attempt(job, stage, source, epochs=1):
    attempt = job.directory / "attempts" / f"{stage}-{uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    output = attempt / "output"
    output.mkdir()
    references = [job.project_dir / "project.json", job.manifest_path, job.directory / "reference.json"]
    reference_list = attempt / "host-private-files.json"
    reference_list.write_text(json.dumps([str(path) for path in references if path.is_file()]), encoding="utf-8")
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
               str(ROOT / "run_generic_job.ps1"), "-Stage", stage, "-Input", str(source), "-Destination", str(output),
               "-ProtectedReferences", str(reference_list)]
    if stage == "train":
        command += ["-Epochs", str(epochs)]
    log = attempt / "launcher.log"
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f"斷網 {stage} 工作失敗。執行紀錄：{log}")
    proof = json.loads((output / "isolation.json").read_bytes())
    manifest = source / ("manifest.json" if stage == "infer" else "dataset.json")
    if (proof.get("stage") != stage or proof.get("manifest_sha256") != digest(manifest)
            or proof.get("appcontainer_token") is not True or proof.get("os_network_capabilities") != 0
            or proof.get("network_denied") is not True or proof.get("labels_denied") is not True
            or proof.get("cuda") is not True):
        raise RuntimeError(f"無法核實 {stage} 的 AppContainer 與 CUDA 憑據：{output}")
    return output


def infer_job(job):
    if job.inference:
        return job
    with TemporaryDirectory(prefix=".inference-input-", dir=job.directory) as directory:
        staged = Path(directory)
        stage_inference(job, staged)
        output = _attempt(job, "infer", staged)
        job.publish_inference(output / "inference.json")
    return job


def _latest_worker_event(path):
    with path.open("rb") as stream:
        end = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, end - 32768))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        if not line.startswith(b"{"):
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(record, dict) and record.get("event") in {"progress", "complete"}:
            return record
    return {}


def job_progress(job, stage, since_ns):
    """Read only the current job's live status from a launcher-owned AppContainer run."""
    local = os.environ.get("LOCALAPPDATA")
    if not local or stage not in ACTIVE_PROFILES:
        return None
    try:
        candidates = [(path.stat().st_mtime_ns, path) for path in
                      (job.directory / "attempts").glob(f"{stage}-*/output/live.json")]
        recent = [item for item in candidates if item[0] >= since_ns]
        if not recent:
            return None
        descriptor = json.loads(max(recent)[1].read_bytes())
        name = descriptor.get("run_folder", "")
        if descriptor.get("stage") != stage or not isinstance(name, str) or not re.fullmatch(r"run-[a-f0-9]{32}", name):
            return None
        base = (Path(local) / "Packages" / ACTIVE_PROFILES[stage] / "AC/Temp").resolve()
        folder = (base / name).resolve()
        if folder.parent != base:
            return None
        if stage == "train":
            report = json.loads((folder / "training/training.json").read_bytes())
            done, total = report.get("epochs_completed"), report.get("epochs_requested")
            if type(done) is int and type(total) is int and 0 <= done <= total and total > 0:
                if done == total:
                    return f"YOLO {done}／{total} 輪已完成，正在準備評估結果。"
                return f"正在從零訓練 YOLO · {done}／{total} 輪"
            return None
        event = _latest_worker_event(folder / "worker.log")
        if event.get("event") == "complete":
            return "批次推論完成，正在核對逐圖結果。"
        if event.get("stage") == "task_preparation":
            return "Qwen 正在準備本批類別的找框提示。"
        phase = INFERENCE_PHASES.get(event.get("stage"))
        done, total = event.get("completed"), event.get("total")
        if phase and type(done) is int and type(total) is int and 0 <= done <= total and total > 0:
            return f"{phase} · {done}／{total} 張"
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return None


def train_job(job, epochs):
    dataset = job.export_dataset()
    output = _attempt(job, "train", dataset, epochs)
    receipt = output / "training" / "training.json"
    evidence = json.loads(receipt.read_bytes())
    if (evidence.get("scope") != "generic_project" or evidence.get("initialization") != "random"
            or evidence.get("pretrained") is not False or evidence.get("resume") is not False
            or evidence.get("dataset_manifest_sha256") != digest(dataset / "dataset.json")):
        raise RuntimeError(f"訓練收據與本次專案資料不符：{receipt}")
    pointer = job.directory / "latest-training.json"
    pending = pointer.with_suffix(".tmp")
    pending.write_text(json.dumps({"report": receipt.relative_to(job.directory).as_posix(),
                                   "report_sha256": digest(receipt),
                                   "dataset": dataset.relative_to(job.directory).as_posix()}), encoding="utf-8")
    pending.replace(pointer)
    return job


def latest_training(job):
    pointer = job.directory / "latest-training.json"
    if not pointer.is_file():
        return None, None, None
    info = json.loads(pointer.read_bytes())
    report = (job.directory / info["report"]).resolve(strict=True)
    dataset = (job.directory / info["dataset"]).resolve(strict=True)
    if (not report.is_relative_to(job.directory / "attempts") or not dataset.is_relative_to(job.directory / "datasets")
            or digest(report) != info["report_sha256"]):
        raise ValueError("訓練紀錄的位置或雜湊不符。")
    return json.loads(report.read_bytes()), report.parent, dataset
