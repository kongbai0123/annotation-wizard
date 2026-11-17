"""Local, text-conditioned proposals for the approved MM Grounding DINO checkpoint."""

from copy import deepcopy
from hashlib import file_digest
import json
import os
from pathlib import Path

from PIL import Image


MODEL_DIR = Path(__file__).resolve().parent.parent / "MM-Grounding-DINO-Tiny"
FILES = {
    "model.safetensors": "2d9bdfd0247f167a00e00fffa1dec1ee5d5861fdfb4e27fd8cc2279af6820688",
    "config.json": "cb8ffc83fa083aebe4a72d4714c759568b4001b32e28a1850e8c2f30c1db5923",
    "preprocessor_config.json": "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def restore_heads(model, checkpoint):
    import torch
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name in ("bbox_embed", "class_embed"):
            original = getattr(model, name)
            if len(original) != 6:
                raise RuntimeError(f"DINO 解碼器的 {name} 層數不符。")
            heads = torch.nn.ModuleList()
            for index in range(6):
                prefix = f"{name}.0." if index == 0 else f"model.decoder.{name}.{index}."
                head = deepcopy(original[0])
                parameters = {k[len(prefix):]: source.get_tensor(k) for k in keys if k.startswith(prefix)}
                head.load_state_dict(parameters, strict=True)
                if any(not torch.equal(v, parameters[k]) for k, v in head.state_dict().items()):
                    raise RuntimeError(f"DINO 第 {index + 1} 層 {name} 參數核對失敗。")
                heads.append(head)
            setattr(model, name, heads)
            setattr(model.model.decoder, name, heads)


def load_model(directory, device):
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    for name, expected in FILES.items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"DINO 本機檔案不存在或版本不符：{path}")
    options = {"local_files_only": True, "trust_remote_code": False}
    processor = AutoProcessor.from_pretrained(directory, **options)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        directory, use_safetensors=True, dtype=torch.float32, use_kernels=False, **options)
    restore_heads(model, directory / "model.safetensors")
    return processor, model.to(device).eval()


def detect(processor, model, device, image, prompt):
    import torch
    from torchvision.ops import nms

    values = processor(images=image, text=prompt.rstrip(".") + ".", return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**values)
    result = processor.post_process_grounded_object_detection(
        output, input_ids=values.input_ids, threshold=0.0, text_threshold=0.25,
        target_sizes=[(image.height, image.width)])[0]
    boxes, scores = result["boxes"].float(), result["scores"].float()
    boxes[:, 0::2].clamp_(0, image.width)
    boxes[:, 1::2].clamp_(0, image.height)
    keep = (torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            & (boxes[:, 2] - boxes[:, 0] >= 8) & (boxes[:, 3] - boxes[:, 1] >= 8))
    boxes, scores = boxes[keep], scores[keep]
    order = nms(boxes, scores, 0.65)[:5]
    return [(tuple(box), score) for box, score in zip(boxes[order].tolist(), scores[order].tolist(), strict=True)]


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class GroundingEngine:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get("ANNOTATION_WIZARD_DINO_DIR", MODEL_DIR)).resolve()
        self.model = None

    def _check_examples(self, processor, model, device):
        folder = self.directory / "health"
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"缺少 DINO 健全性測試圖片：{manifest}")
        examples = json.loads(manifest.read_text(encoding="utf-8"))
        if {row["prompt"] for row in examples} != {"person", "car"}:
            raise ValueError("DINO 健全性測試必須包含 person 和 car。")
        for row in examples:
            path = folder / row["file"]
            if path.resolve().parent != folder or digest(path) != row["sha256"]:
                raise ValueError("DINO 健全性測試圖片的路徑或雜湊不符。")
            with Image.open(path) as source:
                image = source.convert("RGB")
            if not any(overlap(box, row["xyxy"]) >= 0.5
                       for box, _ in detect(processor, model, device, image, row["prompt"])):
                raise RuntimeError(f"DINO 健全性測試未找到 {row['prompt']}：{path.name}")

    def propose(self, image, prompt):
        if self.model is None:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            processor, model = load_model(self.directory, device)
            self._check_examples(processor, model, device)
            self.processor, self.model, self.device = processor, model, device
        return detect(self.processor, self.model, self.device, image, prompt)
