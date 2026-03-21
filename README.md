# Basic-AI-models-Collection
> A collection of fundamental deep learning models implemented from scratch using PyTorch (and pure NumPy). Fully annotated with study notes, designed for deep learning beginners to practice and understand core concepts.

---

## ✨ Key Features
- **Beginner-Friendly**: All models are built from the ground up (no over-encapsulated "black boxes") to help understand underlying principles
- **Clear Learning Path**: Well-organized structure and step-by-step implementation for smooth, stress-free learning
- **Detailed Annotations**: Code includes comprehensive study notes, recording key ideas and common pitfalls
- **Rich Model Coverage**: Covers classic foundational models like MLP, CNN, ResNet, UNet, and DDPM
- **Out-of-the-Box**: Provides ready-to-run training scripts with minimal configuration required

---

## 📁 Directory Structure
```
Basic-AI-models-Collection/
├── myMLP/          # Multi-Layer Perceptron (MNIST handwritten digit classification)
├── myCNN/          # Convolutional Neural Network (image classification task)
├── myResNet/       # Residual Network (solves deep network degradation problem)
├── myUNet/         # U-Net (foundation for medical image segmentation)
├── DDPM/           # Denoising Diffusion Probabilistic Model (foundation for generative AI)
├── data/           # Automatic dataset download and storage directory
├── requirements.txt # Environment dependency list
└── README.md
```
> 💡 **Tip**: For a smooth learning journey, it is recommended to explore the models in the order shown above.
---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/[Your-Username]/Basic-AI-models-Collection.git
cd Basic-AI-models-Collection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
> Core dependencies: `torch`, `numpy`, `matplotlib`, `torchvision`

### 3. Run Examples
Take CNN training as an example:
```bash
cd myCNN
python train.py
```
> For other models, follow the same pattern: enter the corresponding directory and execute `train.py`.

---

## 📖 Learning Guide
For optimal learning progress, we recommend following this order:
1. **myMLP**: Understand fully connected networks, backpropagation, and gradient descent
2. **myCNN**: Learn core computer vision concepts (convolution, pooling, receptive field)
3. **myResNet**: Master residual connections and deep network training techniques
4. **myUNet**: Explore encoder-decoder architecture and skip connections
5. **DDPM**: Get started with diffusion models and understand generative AI fundamentals

Each model directory contains detailed annotations. You can:
- First run the code to observe results
- Then read line-by-line annotations to understand principles
- Try modifying hyperparameters (e.g., learning rate, number of layers) to observe changes

---

## 🤝 Contribution Guidelines
Contributions of any kind are welcome! If you find bugs or have improvement ideas:
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
> You are free to use, modify, and distribute the code in this project, as long as you retain the original copyright notice.

---

## 💬 Contact
If you have questions or suggestions, feel free to reach out via GitHub Issues or email: [your-email@example.com].

---

### 📝 Usage Notes
1. Replace `[Your-Username]` with your actual GitHub username (in the "Clone the Repository" section)
2. Replace `[your-email@example.com]` with your real email address (in the "Contact" section)
3. If you add a LICENSE file to your repository, make sure the license link works (you can create a simple MIT LICENSE file via GitHub's repository settings)
4. This README follows standard GitHub Markdown syntax and will be automatically rendered with clean formatting on GitHub
