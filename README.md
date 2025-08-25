# CNN Models

Welcome to the **CNN Models** repository! This repo contains implementations of popular Convolutional Neural Network (CNN) architectures, designed for learning and experimentation purposes.
All models will be deployed on Hugging Face for easy access and use. You can check out the first model, **AlexNet**, here: [TanishRajput/Alexnet](https://huggingface.co/TanishRajput/Alexnet), **VGG-16**, here: [Tanishrajput/VGG-16](https://huggingface.co/Tanishrajput/VGG-16). More model links will be added soon!

---

## 🚀 Overview

Convolutional Neural Networks (CNNs) are a class of deep learning models particularly effective for image-related tasks such as classification, detection, and segmentation. This repository provides clean, well-documented implementations of famous CNN architectures built using TensorFlow/Keras.

---

## 📚 Included Models

| Model Name  | Description                                         | Dataset(s) Used        | Notes                                   |
|-------------|-----------------------------------------------------|-------------------------|-----------------------------------------|
| AlexNet     | One of the first deep CNNs, popularized deep learning for images. | CIFAR-10, ImageNet    | Adapted for CIFAR-10 with smaller FC layers. |
| VGG16       | Deep CNN with very small (3x3) convolution filters. | CIFAR-10, ImageNet    | Achieved **94.2% accuracy** on CIFAR-10 with custom training. |
<!-- | ResNet50    | Introduced residual connections to combat vanishing gradients. | CIFAR-10, ImageNet    | Powerful architecture for deep networks. | -->
<!-- | MobileNetV2 | Efficient and lightweight CNN architecture for mobile and embedded devices. | CIFAR-10, ImageNet    | Great for resource-constrained environments. | -->

*(More models coming soon!)*

---

## 🔗 Live Demos

You can try the deployed AlexNet model here:

- 🌐 **Streamlit App:** [Link](https://cnn-models-dhuwwjq428nsujvkbmrvd3.streamlit.app)

You can try the deployed VGG-16 model here:

- 🌐 **Streamlit App:** [Link](https://cnn-models-7jb2etgepabdragvfhtdpr.streamlit.app)

--- 

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/tanishra/CNN-Models.git
   cd CNN-Models
   `````
2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ````
3. Install dependencies for a specific model:
    ### AlexNet
    ```bash
    cd alexnet
    pip install -r requirements.txt
    ```

    ### VGG-16
    ```bash
    cd vgg-16
    pip install -r requirements.txt
    `````
4. Run the model scripts or Streamlit apps according to the folder you choose.
    ```bash
    streamlit run app.py
    `````

---

## 🛠 Usage
- Each model folder contains:
- Model code: Implementation of the CNN architecture.
- Training script: Code to train the model on datasets like CIFAR-10.
- Evaluation script: Test the trained model on validation or test sets.

---

## 🤝 Contribution
Contributions are welcome! Feel free to:
- Add more CNN architectures.
- Improve existing implementations.
- Add tutorials or notebooks demonstrating usage.
Please fork the repo and create a pull request.

---

<!-- ## 📄 License
This repository is licensed under the MIT License. See the LICENSE file for details. -->
