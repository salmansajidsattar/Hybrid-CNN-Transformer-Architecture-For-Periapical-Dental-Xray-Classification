import torch
from collections import OrderedDict
import subprocess
from config import Config
# ---------------------------
# 1. IMPORT YOUR MODEL
# ---------------------------
from model import create_model      # ⚠️ change to your file name

CKPT_PATH = Config.PROJECT_ROOT/ "checkpoints/best_model.pth"                # ⚠️ change if needed
ONNX_PATH = "model.onnx"


def load_checkpoint(model, ckpt_path):
    """
    Safely loads PyTorch checkpoint and handles:
    - model_state_dict extraction
    - DataParallel 'module.' prefix
    - strict vs non-strict loading
    """
    print(f"\n📥 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # If checkpoint has full dict, extract model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if exists
    new_state = OrderedDict()
    for k, v in state_dict.items():
        name = k[len("module."):] if k.startswith("module.") else k
        new_state[name] = v

    # Try strict first
    try:
        model.load_state_dict(new_state, strict=True)
        print("✅ Loaded weights (strict=True)")
    except RuntimeError as e:
        print("⚠️ Strict load failed → switching to strict=False")
        print(e)
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print("\nMissing keys:", missing)
        print("Unexpected keys:", unexpected)

    return model


def export_onnx(model, onnx_path):
    print("\n📦 Exporting ONNX...")

    dummy_input = torch.randn(1, 3, 224, 224)  # match model input

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    print(f"✅ ONNX saved: {onnx_path}")


def open_netron(onnx_path):
    print("\n🌐 Opening Netron...")
    try:
        subprocess.Popen(["netron", onnx_path])
        print("If browser doesn’t open automatically, open https://netron.app and drop the model.")
    except FileNotFoundError:
        print("⚠️ Netron is not installed. Install using: pip install netron")


if __name__ == "__main__":
    print("========== MODEL VISUALIZER ==========")

    # ---------------------------
    # 2. Create model
    # ---------------------------
    model = create_model(num_classes=2)   # ⚠️ edit hyperparams if needed
    model.eval()

    # ---------------------------
    # 3. Load checkpoint
    # ---------------------------
    model = load_checkpoint(model, CKPT_PATH)

    # ---------------------------
    # 4. Export ONNX
    # ---------------------------
    export_onnx(model, ONNX_PATH)

    # ---------------------------
    # 5. Launch Netron
    # ---------------------------
    open_netron(ONNX_PATH)

    print("\n🎉 Visualization complete!")
