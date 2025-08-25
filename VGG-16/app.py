import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# VGG config
cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']

# Create VGG architecture with BatchNorm
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def make_layers(cfg_list, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg_list:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16_bn(num_classes=10):
    return VGG(make_layers(cfg_vgg16, batch_norm=True), num_classes=num_classes)

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = vgg16_bn(num_classes=10)
model.load_state_dict(torch.load("VGG_model.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.title("CIFAR-10 Image Classifier (VGG16)")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 classes.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.subheader(f"Predicted Class: **{CLASSES[predicted.item()]}**")
    st.write(f"Confidence: **{confidence.item()*100:.2f}%**")

    # Show top 5 predictions
    st.subheader("Top 5 Predictions:")
    top5_prob, top5_classes = torch.topk(probs, 5)
    for i in range(5):
        st.write(f"{CLASSES[top5_classes[0][i]]}: {top5_prob[0][i]*100:.2f}%")
