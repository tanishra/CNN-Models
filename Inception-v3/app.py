import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision.datasets import CIFAR100
from collections import OrderedDict


# CIFAR-100 class labels
cifar100_classes = CIFAR100(root="./data", train=False, download=True).classes

# Model Loader
@st.cache_resource
def load_model():
    # Download model file from Hugging Face Hub
    checkpoint_path = hf_hub_download(
        repo_id="TanishRajput/Inception-v3",
        filename="Inception-v3.pt"
    )

    # Define Inception-v3 architecture (disable aux logits for inference)
    model = models.inception_v3(pretrained=False, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel "module." prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# Preprocessing pipeline
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


# Streamlit UI
st.set_page_config(page_title="Inception-v3 CIFAR-100 Classifier", layout="centered")
st.title("🔍 CIFAR-100 Image Classifier (Inception-v3)")

st.write("Upload an image and let the trained Inception-v3 model predict its CIFAR-100 class.")

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("🔮 Prediction Results")
    for i in range(5):
        st.write(f"**{cifar100_classes[top5_idx[i]]}**: {top5_prob[i].item()*100:.2f}%")
