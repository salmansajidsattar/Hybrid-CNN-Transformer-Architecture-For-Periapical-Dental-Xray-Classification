# Dental Periapical Disease Classification

A deep learning system for automated detection and classification of periapical diseases from dental radiographs using a hybrid CNN-Transformer architecture.

## 🦷 Overview

This project implements a state-of-the-art deep learning model for classifying dental X-ray images into periapical and non-periapical disease categories. The system combines Convolutional Neural Networks (CNN) for feature extraction with Transformer encoders for enhanced pattern recognition.

## 🏗️ Architecture

### Model Pipeline

The model processes dental radiographs through the following stages:

1. **Input Image** - Accepts dental X-ray images (B, 3, 384, 384)
2. **CNN Feature Extraction** - Extracts spatial features (B, 512, 24, 24)
3. **Patch Embedding** - Converts feature maps to sequence tokens (B, 576, 512)
4. **CLS Token Addition** - Adds classification token (B, 576, 512)
5. **Positional Encoding** - Encodes spatial information (B, 576, 512)
6. **Transformer Encoder** - Processes sequences for global context (B, 576, 512)
7. **CLS Token Extraction** - Extracts learned representation (B, 512)
8. **Classification Head** - Binary classification output (B, 2)

![Architecture Diagram](https://github.com/salmansajidsattar/Hybrid-CNN-Transformer-Architecture-For-Periapical-Dental-Xray-Classification/blob/main/Architecture_diagram/Pipeline.drawio.png)

### Hybrid CNN-Transformer Design

- **CNN Backbone**: Efficient spatial feature extraction from radiographic images
- **Transformer Encoder**: Captures long-range dependencies and contextual relationships
- **CLS Token**: Aggregates global image representation for classification

## 📊 Dataset

The model is trained on the **Mendeley Dental Dataset** containing:
- Periapical disease X-rays
- Non-periapical (healthy) X-rays
- Various imaging conditions and anatomical variations

## 🚀 Project Structure

```
dental-periapical-detection/
├── model.py              # CNN-Transformer architecture definition
├── train.py              # Training script
├── evaluate.py           # Model evaluation and metrics
├── streamlit_ui/         # Interactive web interface
├── data/
│   ├── periapical/       # Periapical disease images
│   └── non-periapical/   # Healthy dental images
├── checkpoints/          # Saved model weights
└── results/              # Training curves and predictions
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dental-periapical-detection.git
cd dental-periapical-detection

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage
### Pre-trained Model

Download the pre-trained model weights:

**[Download Model (Google Drive)](https://drive.google.com/file/d/13Hx3y25r5ZPexlyxVKDlcuHYg-mpLZbh/view?usp=drive_link)**

Place the downloaded model in the `checkpoints/` directory:

```bash
mkdir -p checkpoints
# Move downloaded model to checkpoints/
mv ~/Downloads/best_model.h5 checkpoints/
```


### Training

```bash
python train.py --data_dir ./data \
                --epochs 100 \
                --batch_size 16 \
                --learning_rate 0.001 \
                --checkpoint_dir ./checkpoints
```

### Evaluation

```bash
python evaluate.py --model_path ./checkpoints/best_model.h5 \
                   --test_dir ./data/test \
                   --output_dir ./results
```

### Interactive Interface

Launch the Streamlit web application for real-time predictions:

```bash
streamlit run streamlit_ui/app.py
```

Upload dental X-rays through the web interface and get instant classification results with confidence scores.

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 88.6% |
| Precision | 89.5% |
| Recall | 89.5% |
| F1-Score | 89.5% |

### Training Curves

- Training and validation loss curves
- Accuracy progression over epochs
- Confusion matrix visualization

![Training Results](https://github.com/salmansajidsattar/Hybrid-CNN-Transformer-Architecture-For-Periapical-Dental-Xray-Classification/blob/main/results/training_history.png)

## 🔍 Key Features

- **Hybrid Architecture**: Combines CNN's local feature extraction with Transformer's global attention
- **Patch-based Processing**: Efficient handling of high-resolution medical images
- **K-Nearest Neighbors Integration**: Additional similarity-based validation
- **Real-time Inference**: Interactive web interface for clinical deployment
- **Visualization Tools**: Attention maps and feature visualization for interpretability

## 📝 Model Details

### CNN Feature Extractor
- Input resolution: 384×384×3
- Output feature maps: 512 channels
- Spatial dimensions: 24×24

### Transformer Configuration
- Number of encoder layers: 6-12 (configurable)
- Attention heads: 8
- Hidden dimension: 512
- Feedforward dimension: 2048

### Classification Head
- Input: 512-dimensional CLS token
- Output: 2 classes (Periapical / Non-Periapical)
- Activation: Softmax

## 🎯 Use Cases

- **Clinical Diagnosis Support**: Assist dentists in identifying periapical lesions
- **Screening Tool**: Rapid preliminary assessment of dental radiographs
- **Educational Tool**: Training dental students in radiographic interpretation
- **Research**: Dataset for dental AI research and development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{dental_periapical_2024,
  title={Automated Periapical Disease Detection using CNN-Transformer Architecture},
  author={SALMAN SAJID & HASSAN SARDAR},
  year={2025}
}
```

## 🙏 Acknowledgments

- Mendeley Dental Dataset contributors
- TensorFlow and Keras development teams
- Open-source community

## 📧 Contact

For questions or collaboration opportunities, please reach out:
- Email: salmansajidsattar@gmail.com
- GitHub: [@salmansajidsattar](https://github.com/salmansajidsattar)
- LinkedIn: [salmansajidsattar](https://linkedin.com/in/salmansajidsattar)

## 🔮 Future Work

- [ ] Multi-class classification for different periapical conditions
- [ ] Integration with DICOM standards
- [ ] Mobile application deployment
- [ ] Explainable AI visualisation improvements
- [ ] Real-time video stream processing
- [ ] Multi-modal fusion with patient metadata

---

**Note**: This is a research project for educational and development purposes. Always consult qualified dental professionals for medical diagnoses.
