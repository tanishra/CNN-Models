# CNN Models

Welcome to the **CNN Models** repository! This repo contains implementations of popular Convolutional Neural Network (CNN) architectures, designed for learning and experimentation purposes.
All models will be deployed on Hugging Face for easy access and use. 

You can check out the models here:  
- **AlexNet**: [TanishRajput/Alexnet](https://huggingface.co/TanishRajput/Alexnet)  
- **VGG-16**: [TanishRajput/VGG-16](https://huggingface.co/TanishRajput/VGG-16)  
- **Inception-v1 (GoogLeNet)**: [TanishRajput/Inception-v1](https://huggingface.co/TanishRajput/Inception-v1) 

 More model links will be added soon!

---

## 🚀 Overview

Convolutional Neural Networks (CNNs) are a class of deep learning models particularly effective for image-related tasks such as classification, detection, and segmentation. This repository provides clean, well-documented implementations of famous CNN architectures built using TensorFlow/Keras.

---

## 📚 Included Models

| Model            | Key Idea                                           | Dataset        | Performance                                  |
|------------------|---------------------------------------------------|---------------|---------------------------------------------|
| **AlexNet**      | A pioneering CNN that popularized deep learning. | CIFAR-10      | Optimized for small image sizes.           |
| **VGG-16**       | Deep network with uniform 3×3 convolution layers.| CIFAR-10      | Reached **94.2% accuracy** with tuning.    |
| **Inception-v1** | Multi-scale feature extraction using Inception modules. | CIFAR-10 | Achieved **91.21% test accuracy** with custom training. |
<!-- | ResNet50    | Introduced residual connections to combat vanishing gradients. | CIFAR-10, ImageNet    | Powerful architecture for deep networks. | -->
<!-- | MobileNetV2 | Efficient and lightweight CNN architecture for mobile and embedded devices. | CIFAR-10, ImageNet    | Great for resource-constrained environments. | -->

*(More models coming soon!)*

---

## 🔗 Live Demos

You can try the deployed models here:

- **AlexNet**:  
- 🌐 **Streamlit App:** [Link](https://cnn-models-dhuwwjq428nsujvkbmrvd3.streamlit.app)

- **VGG-16**:  
- 🌐 **Streamlit App:** [Link](https://cnn-models-7jb2etgepabdragvfhtdpr.streamlit.app)

- **Inception-v1 (GoogLeNet)**:  
  🌐 **Streamlit App:** [Link](https://cnn-models-phf53k6qqgkw4blwzql85r.streamlit.app)

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
    ```

    ### Inception-v1
    ```bash
    cd Inception-v1
    pip install -r requirements.txt
    ````

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
