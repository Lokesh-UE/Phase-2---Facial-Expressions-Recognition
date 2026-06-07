# Facial Expression Recognition using Deep CNN and Transfer Learning on FER2013

A complete deep learning pipeline for classifying facial expressions from the FER2013 dataset. Trains a custom CNN from scratch and compares it against VGG16 and ResNet50 transfer learning models.

**Best model:** Custom CNN — **64.47%** test accuracy

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Test Loss |
|-------|----------|-----------|--------|----------|-----------|
| **Custom CNN** | **64.47%** | **61.10%** | **64.27%** | **61.91%** | **0.970** |
| VGG16 | 49.51% | 46.05% | 50.82% | 46.61% | 1.354 |
| ResNet50 | 32.81% | 28.26% | 31.17% | 26.80% | 1.709 |

### Per-Class Results (Custom CNN)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Angry | 0.54 | 0.61 | 0.57 | 958 |
| Disgust | 0.50 | 0.71 | 0.59 | 111 |
| Fear | 0.54 | 0.38 | 0.44 | 1,024 |
| Happy | 0.89 | 0.83 | 0.86 | 1,774 |
| Sad | 0.55 | 0.44 | 0.49 | 1,247 |
| Surprise | 0.72 | 0.82 | 0.77 | 831 |
| Neutral | 0.54 | 0.70 | 0.61 | 1,233 |

**Key takeaways:**
- The custom CNN significantly outperformed both transfer learning models on this small grayscale dataset.
- Happy was the easiest class (F1 = 0.86) due to its large sample size and distinctive smile.
- Fear was the most difficult (recall = 0.38), often confused with Sad and Surprise.
- Transfer learning from ImageNet struggled because FER2013 images are tiny (48×48), grayscale, and domain-specific.

---

## Dataset

**FER2013** — [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

| Property | Value |
|----------|-------|
| Total images | 35,887 |
| Training images | 28,709 |
| Test images | 7,178 |
| Image size | 48 × 48 pixels |
| Image type | Grayscale |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |

The dataset is highly imbalanced — Happy has 7,215 training samples while Disgust has only 436. Class weights and data augmentation were used to address this.

---

## Models

### Custom CNN

Designed specifically for 48×48 grayscale face images:

| Layer | Configuration |
|-------|---------------|
| Input | 48 × 48 × 1 |
| Conv Block 1 | Conv2D(32) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25) |
| Conv Block 2 | Conv2D(64) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25) |
| Conv Block 3 | Conv2D(128) × 2, BatchNorm, MaxPool(2×2), Dropout(0.25) |
| Classifier | Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(7, Softmax) |

**Training:** Adam optimizer, lr=0.001, batch size 64, up to 50 epochs with early stopping (patience=10) and learning rate reduction on plateau.

### Transfer Learning Models

- **VGG16:** ImageNet pre-trained weights, frozen base + custom classifier head, fine-tuned top 4 blocks.
- **ResNet50:** Same setup, fine-tuned top 10 layers.

Both received grayscale images replicated to 3 channels and the same class weights as the custom CNN.

---

## Project Structure

```
facial_expression_recognition/
├── run_project.py                        # Main script — run full pipeline locally
├── facial_expression_recognition.py      # Same code (alternate name)
├── facial_expression_recognition.ipynb   # Self-contained Kaggle notebook
├── requirements.txt
├── README.md                             # This file
├── outputs/                              # Generated after training
│   ├── 01_sample_images.png
│   ├── 02_class_distribution.png
│   ├── 03_cnn_training_history.png
│   ├── 04_cnn_confusion_matrix.png
│   ├── 05_cnn_roc_curves.png
│   ├── 06_vgg_training_history.png
│   ├── 06_vgg_confusion_matrix.png
│   ├── 07_vgg_roc_curves.png
│   ├── 08_resnet_training_history.png
│   ├── 08_resnet_confusion_matrix.png
│   ├── 09_resnet_roc_curves.png
│   ├── 10_model_comparison.png
│   ├── 11_error_analysis.png
│   ├── model_comparison.csv
│   ├── gradcam_cnn/
│   └── *_classification_report.txt
└── checkpoints/                          # Saved model weights
    ├── custom_cnn_final.keras
    ├── vgg16_final.keras
    └── resnet50_final.keras
```

---

## How to Run

### On Kaggle (Recommended)

1. Upload `facial_expression_recognition.ipynb` to [Kaggle](https://www.kaggle.com/code) (or copy it into a new notebook)
2. Add Data → search **FER2013** → add `msambare/fer2013`
3. Settings → Accelerator → **GPU T4 x2**
4. **Run All** cells — the notebook is self-contained (no extra file uploads needed)

The notebook auto-detects the dataset and runs the full pipeline on all three models. Set `Config.QUICK_DEMO = True` in the run cell for a fast smoke test.

### Locally

```bash
pip install -r requirements.txt
```

**Option A — CSV file:**
```bash
python run_project.py --csv ./fer2013/fer2013.csv
```

**Option B — Image folders:**
```bash
python run_project.py --data-dir ./fer2013/
```

**Quick demo:**
```bash
python run_project.py --demo --data-dir ./fer2013/
```

---

## Dependencies

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
```

---

## Evaluation Outputs

The pipeline automatically generates:

- Training/validation accuracy and loss curves
- Normalized confusion matrices for all models
- ROC curves with per-class AUC (one-vs-rest)
- Per-class classification reports
- Grad-CAM heatmaps for model interpretability
- Model comparison bar chart and CSV table
- Error analysis on misclassified samples

---

## What Was Learned

- A carefully designed custom CNN can outperform generic transfer learning when the target domain (small grayscale faces) differs significantly from the pre-training domain (large RGB natural images).
- Data augmentation and class weighting are essential for handling severe class imbalance.
- Grad-CAM confirmed the model focuses on the eyes and mouth — the most emotionally expressive facial regions.
- Fear vs. Sad vs. Surprise is the hardest distinction, even for humans.

---

## References

- Goodfellow et al. (2013). FER2013 Dataset. [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Simonyan & Zisserman (2014). VGG Networks. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He et al. (2016). ResNet. [CVPR 2016](https://doi.org/10.1109/CVPR.2016.90)
- Selvaraju et al. (2017). Grad-CAM. [ICCV 2017](https://doi.org/10.1109/ICCV.2017.74)
