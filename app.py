"""
Brain Tumor Classifier - Gradio App
"""
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import gradio as gr
import yaml
from PIL import Image

from src.models.classifier import TumorClassifier
from src.data.dataset import get_transforms
from src.explainability.gradcam import GradCAMVisualizer
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Load config & model ───────────────────────────────────────

CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoints" / "model_best.pth"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config["data"]["classes"]
IMAGE_SIZE = config["data"]["image_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TumorClassifier(
    backbone=config["model"]["backbone"],
    num_classes=config["model"]["num_classes"],
    pretrained=False,
)

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

transform = get_transforms("val", IMAGE_SIZE)

explainer = GradCAMVisualizer(
    model=model,
    class_names=CLASS_NAMES,
    device=device,
    method=config["explainability"]["method"],
)

# ── Inference function ────────────────────────────────────────

def predict(image: np.ndarray):
    if image is None:
        return None, "Please upload an image."

    augmented = transform(image=image)
    tensor = augmented["image"]
    result = explainer.explain_single_tensor(tensor)

    probs = result["all_probs"]
    prob_output = {cls: round(prob * 100, 2) for cls, prob in probs.items()}

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = tensor.numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1).astype(np.float32)
    overlay = show_cam_on_image(img_np, result["heatmap"], use_rgb=True)

    label = (
        f"**Prediction: {result['predicted_class'].upper()}**\n"
        f"Confidence: {result['confidence']*100:.1f}%"
    )

    return overlay, label, prob_output

# ── Gradio UI ─────────────────────────────────────────────────

with gr.Blocks(title="AI Tumor Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 AI Brain Tumor Classifier
    Upload a brain MRI scan to classify the tumor type and visualize what the model focuses on using **Grad-CAM**.

    > ⚠️ **Disclaimer**: This is a research portfolio project and is NOT intended for clinical use.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload MRI Scan", type="numpy")
            submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=1):
            cam_output = gr.Image(label="Grad-CAM Heatmap (what the model sees)")
            label_output = gr.Markdown(label="Prediction")
            prob_output = gr.Label(label="Class Probabilities", num_top_classes=4)

    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[cam_output, label_output, prob_output],
    )

    gr.Markdown("""
    ---
    **Classes:** Glioma | Meningioma | Pituitary Tumor | No Tumor

    **Model:** EfficientNetB3 fine-tuned on Brain Tumor MRI Dataset  
    **Explainability:** Grad-CAM++ highlights regions influencing the prediction
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)