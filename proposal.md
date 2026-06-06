# Facial Expression Recognition using CNN and Transfer Learning

**Machine Learning — Phase 2: Proposal and Code Implementation**

---

## 1. Project Title

Facial Expression Recognition using Deep CNN and Transfer Learning on FER2013

---

## 2. Background and Motivation

Facial expressions are a natural way for people to show emotions. Systems that can read these expressions automatically are useful in human-computer interaction, healthcare, education, and security. Traditional machine learning methods struggle with variation in faces, lighting, and pose. Convolutional Neural Networks (CNNs) learn features directly from images and have become the standard approach for facial expression recognition (FER).

This project implements a full FER pipeline on the FER2013 dataset. A custom CNN is built and trained from scratch, then compared with two transfer learning models (VGG16 and ResNet50) to find which approach works best for classifying emotions from grayscale face images.

---

## 3. Problem Statement

The goal is to classify 48×48 grayscale face images into seven emotion categories:

**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**

The project must:
- Load and preprocess the FER2013 dataset
- Train a custom CNN architecture
- Apply transfer learning with VGG16 and ResNet50
- Evaluate all models using standard metrics and visualizations
- Compare model performance and analyse errors

**Challenges:**
- Class imbalance (Disgust has far fewer samples than Happy)
- Similar-looking emotions (Fear vs Surprise, Sad vs Neutral)
- Small image size (48×48) limits fine detail
- Transfer learning models were trained on large RGB ImageNet images, which differ from FER2013

---

## 4. Selected Dataset and Link

**Dataset:** FER2013 (Facial Expression Recognition 2013)

**Link:** https://www.kaggle.com/datasets/msambare/fer2013

---

## 5. Dataset Description

| Property | Value |
|----------|-------|
| Total images | 35,887 |
| Training images | 28,709 |
| Test images | 7,178 |
| Image size | 48 × 48 pixels |
| Image type | Grayscale |
| Number of classes | 7 |
| Format used | Folder structure (`train/` and `test/` with emotion subfolders) |

### Class Labels

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

### Class Distribution (Training Set)

| Emotion | Count |
|---------|-------|
| Angry | 3,995 |
| Disgust | 436 |
| Fear | 4,097 |
| Happy | 7,215 |
| Sad | 4,830 |
| Surprise | 3,171 |
| Neutral | 4,965 |

Disgust is heavily underrepresented. This was handled using class-weighted loss and data augmentation during training.

---

## 6. Research Questions

1. Can a custom CNN achieve competitive accuracy on FER2013 without pre-trained weights?
2. How does data augmentation affect model training and generalization?
3. Which facial regions does the model use for prediction (Grad-CAM analysis)?
4. Do VGG16 and ResNet50 outperform the custom CNN after fine-tuning?
5. Which emotion pairs are most frequently confused?

---

## 7. Proposed Methodology

### 7.1 Data Loading and Split

- Images loaded from Kaggle folder structure: `train/` and `test/` with seven emotion subfolders
- Training set split into 80% train and 20% validation (stratified)
- Test set used for final evaluation (7,178 images)

### 7.2 Preprocessing

- Images resized to 48×48 (already correct size)
- Pixel values normalized to [0, 1] by dividing by 255
- Labels converted to one-hot encoding for 7 classes
- Grayscale images replicated to 3 channels for VGG16 and ResNet50

### 7.3 Data Augmentation

Applied to training data only:
- Random horizontal flip
- Random rotation (±15°)
- Random width/height shift (±10%)
- Random zoom (±10%)

### 7.4 Custom CNN Design

| Layer | Configuration |
|-------|---------------|
| Input | 48 × 48 × 1 |
| Conv Block 1 | Conv2D(32) × 2, BatchNorm, MaxPool, Dropout(0.25) |
| Conv Block 2 | Conv2D(64) × 2, BatchNorm, MaxPool, Dropout(0.25) |
| Conv Block 3 | Conv2D(128) × 2, BatchNorm, MaxPool, Dropout(0.25) |
| Classifier | Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(7, Softmax) |

**Training:** Adam optimizer (lr = 0.001), categorical crossentropy with class weights, batch size 64, up to 50 epochs with early stopping.

### 7.5 Transfer Learning Models

**VGG16:**
- Pre-trained on ImageNet, base layers frozen initially
- Custom head: GlobalAveragePooling → Dense(512) → Dropout(0.5) → Softmax(7)
- Fine-tuned top 4 convolutional blocks with lower learning rate

**ResNet50:**
- Same setup as VGG16
- Fine-tuned top 10 layers with lower learning rate

### 7.6 Evaluation Metrics

- Accuracy, precision, recall, F1-score (macro and per-class)
- Confusion matrix (normalized)
- ROC curves with AUC (one-vs-rest)
- Training and validation loss/accuracy curves
- Grad-CAM heatmaps for interpretability
- Error analysis on misclassified samples

### 7.7 Implementation Environment

- **Framework:** TensorFlow 2.19 / Keras
- **Platform:** Kaggle Notebook (GPU T4 × 2)
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, OpenCV

---

## 8. Expected Results

Before running experiments, the expected outcomes were:
- Custom CNN accuracy in the 65–70% range
- Transfer learning models competitive with or better than the custom CNN
- Confusion between Fear/Surprise and Sad/Neutral
- Grad-CAM highlighting eyes and mouth regions
- Complete set of figures and a model comparison table

---

## 9. Actual Results

### 9.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Test Loss |
|-------|----------|-----------|--------|----------|-----------|
| **Custom CNN** | **64.47%** | **61.10%** | **64.27%** | **61.91%** | **0.970** |
| VGG16 | 49.51% | 46.05% | 50.82% | 46.61% | 1.354 |
| ResNet50 | 32.81% | 28.26% | 31.17% | 26.80% | 1.709 |

**Best model: Custom CNN** with 64.47% test accuracy and the lowest test loss.

### 9.2 Per-Class Results (Custom CNN)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.54 | 0.61 | 0.57 |
| Disgust | 0.50 | 0.71 | 0.59 |
| Fear | 0.54 | 0.38 | 0.44 |
| Happy | 0.89 | 0.83 | 0.86 |
| Sad | 0.55 | 0.44 | 0.49 |
| Surprise | 0.72 | 0.82 | 0.77 |
| Neutral | 0.54 | 0.70 | 0.61 |

Happy and Surprise were classified best. Fear had the lowest recall (0.38).

### 9.3 Research Question Answers

1. **Custom CNN without pre-training:** Yes. The custom CNN reached 64.47% accuracy.
2. **Data augmentation:** Augmentation and class weights helped keep training stable.
3. **Grad-CAM:** Heatmaps showed focus on eye and mouth areas.
4. **Transfer learning vs custom CNN:** VGG16 and ResNet50 did not beat the custom CNN.
5. **Confused pairs:** Fear with Sad/Surprise, and Sad with Neutral.

### 9.4 Key Findings

- A task-specific custom CNN outperformed generic pre-trained models on this dataset.
- Happy was the easiest class (F1 = 0.86).
- Fear was the hardest class (F1 = 0.44).
- All required outputs were generated: training curves, confusion matrices, ROC curves, Grad-CAM, error analysis, and model comparison chart.

---

## 10. Generated Figures and Tables

| Output | File |
|--------|------|
| Sample images per class | `01_sample_images.png` |
| Class distribution | `02_class_distribution.png` |
| Custom CNN training curves | `03_cnn_training_history.png` |
| Custom CNN confusion matrix | `04_cnn_confusion_matrix.png` |
| Custom CNN ROC curves | `05_cnn_roc_curves.png` |
| VGG16 training curves | `06_vgg_training_history.png` |
| VGG16 confusion matrix | `06_vgg_confusion_matrix.png` |
| VGG16 ROC curves | `07_vgg_roc_curves.png` |
| ResNet50 training curves | `08_resnet_training_history.png` |
| ResNet50 confusion matrix | `08_resnet_confusion_matrix.png` |
| ResNet50 ROC curves | `09_resnet_roc_curves.png` |
| Model comparison chart | `10_model_comparison.png` |
| Error analysis | `11_error_analysis.png` |
| Model comparison table | `model_comparison.csv` |
| Grad-CAM visualizations | `gradcam_cnn/` |
| Classification reports | `*_classification_report.txt` |

---

## 11. Conclusion

This project successfully built and evaluated a facial expression recognition system on FER2013. The custom CNN achieved 64.47% test accuracy and outperformed both VGG16 and ResNet50. For small grayscale face images, a dedicated CNN architecture was more effective than transfer learning from ImageNet in this setup.

---

## 12. References

1. Goodfellow, I. J., et al. (2013). Challenges in Representation Learning. *Neural Information Processing*, 117–124.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*, 770–778.
4. Selvaraju, R. R., et al. (2017). Grad-CAM. *ICCV*, 618–626.

---

*Phase 2 — Proposal and Code Implementation*
