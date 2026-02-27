"""
app/server.py
──────────────
Flask backend that:
  1. Serves the Neural Diagnostics HTML interface
  2. Accepts MRI image uploads via POST /predict
  3. Runs the trained EfficientNet model
  4. Returns prediction + Grad-CAM as base64 images

Run AFTER training is complete:
    cd ~/Downloads/tumor_classifier
    source venv/bin/activate
    python app/server.py
Then open: http://localhost:5000
"""

import sys
import io
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.classifier import TumorClassifier
from src.data.dataset import get_transforms

# ── Config ────────────────────────────────────────────────────
CONFIG_PATH    = Path(__file__).parent.parent / "configs/config.yaml"
CHECKPOINT     = Path(__file__).parent.parent / "outputs/checkpoints/model_best.pth"
TEMPLATES_DIR  = Path(__file__).parent.parent / "app"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

CLASS_NAMES  = config["data"]["classes"]
IMAGE_SIZE   = config["data"]["image_size"]
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model ────────────────────────────────────────────────
print("Loading model...")
model = TumorClassifier(
    backbone   = config["model"]["backbone"],
    num_classes= config["model"]["num_classes"],
    pretrained = False,
)
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()
print(f"✓ Model loaded (val_accuracy={ckpt.get('val_accuracy', '?'):.4f})")

# ── Grad-CAM setup ────────────────────────────────────────────
def get_target_layer(model):
    name = model.backbone_name.lower()
    if "efficientnet" in name:
        return [model.backbone.conv_head]
    elif "resnet" in name:
        return [list(model.backbone.children())[-3][-1]]
    else:
        for m in reversed(list(model.backbone.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return [m]

cam = GradCAM(model=model, target_layers=get_target_layer(model))

# ── Transforms ────────────────────────────────────────────────
transform = get_transforms("val", IMAGE_SIZE)

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(TEMPLATES_DIR))

@app.route("/")
def index():
    """Serve the Neural Diagnostics UI."""
    return send_from_directory(str(TEMPLATES_DIR), "neural_diagnostics.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data with 'file' field (image)
    Returns: JSON with prediction, confidence, probabilities, base64 images
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        # Read image
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np    = np.array(pil_img)

        # Preprocess
        augmented = transform(image=img_np)
        tensor    = augmented["image"].unsqueeze(0).to(device)  # (1, C, H, W)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx    = int(probs.argmax())
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = round(float(probs[pred_idx]) * 100, 2)

        # Grad-CAM
        targets      = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

        # Reconstruct RGB image for overlay
        mean    = np.array([0.485, 0.456, 0.406])
        std     = np.array([0.229, 0.224, 0.225])
        img_rgb = tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_rgb = std * img_rgb + mean
        img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
        overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Convert images to base64
        def to_b64(arr):
            arr_uint8 = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode("utf-8")

        # Also encode the original resized image
        orig_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))

        return jsonify({
            "success":           True,
            "prediction":        pred_class,
            "confidence":        confidence,
            "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "image":             to_b64(orig_resized),
            "gradcam":           to_b64(overlay),
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Neural Diagnostics Server")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)