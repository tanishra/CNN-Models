import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from collections import OrderedDict


# CIFAR-100 class labels
CIFAR100_CLASSES = [
 'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


# Load Model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="tanishrajput/ResNet-50", filename="ResNet-50.pth")

    # model architecture 
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 has 100 classes

    # Load checkpoint
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle DataParallel case
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # strip "module." 
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

model = load_model()


# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean/std
])


# Streamlit UI
st.title("CIFAR-100 Image Classifier 🖼️")
st.write("Upload an image and let the trained ResNet-50 (from Hugging Face) predict its class!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.topk(probs, 5)

        # Show top-5 predictions
        st.subheader("Predictions")
        for i in range(5):
            st.write(f"{i+1}. {CIFAR100_CLASSES[top_class[0][i]]} ({top_prob[0][i].item()*100:.2f}%)")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
