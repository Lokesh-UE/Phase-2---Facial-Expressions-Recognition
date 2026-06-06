# Facial Expression Recognition

Phase 2 project for Machine Learning course. Classifies facial expressions from the FER2013 dataset using a custom CNN and transfer learning (VGG16, ResNet50).

## Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Custom CNN** | **64.47%** | **61.91%** |
| VGG16 | 49.51% | 46.61% |
| ResNet50 | 32.81% | 26.80% |

Best model: **Custom CNN**

## Project Structure

```
facial_expression_recognition/
├── proposal.md                          # Project proposal
├── facial_expression_recognition.py     # Main training script
├── facial_expression_recognition.ipynb  # Kaggle notebook
├── requirements.txt
├── README.md
├── outputs/                             # Generated after training
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
└── checkpoints/                         # Saved models
    ├── custom_cnn_final.keras
    ├── vgg16_final.keras
    └── resnet50_final.keras
```

## Dataset

**FER2013** — https://www.kaggle.com/datasets/msambare/fer2013

| Property | Value |
|----------|-------|
| Total images | 35,887 |
| Image size | 48 × 48 grayscale |
| Classes | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |

On Kaggle the dataset uses folder structure:
```
train/angry/, train/happy/, ...
test/angry/, test/happy/, ...
```

## Run on Kaggle

1. Create a new notebook on [Kaggle](https://www.kaggle.com/code)
2. Add Data → search **FER2013** → add `msambare/fer2013`
3. Add Data → Upload → `facial_expression_recognition.py`
4. Settings → Accelerator → **GPU T4 x2**
5. Run the notebook:

```python
import os, sys, shutil, importlib

SCRIPT = 'facial_expression_recognition.py'
DEST = f'/kaggle/working/{SCRIPT}'

if not os.path.exists(DEST):
    for root, _, files in os.walk('/kaggle/input'):
        if SCRIPT in files:
            shutil.copy(os.path.join(root, SCRIPT), DEST)
            break

sys.path.insert(0, '/kaggle/working')
import facial_expression_recognition as fer
importlib.reload(fer)

DATA_DIR = '/kaggle/input/datasets/msambare/fer2013'
fer.Config.QUICK_DEMO = False

results = fer.run_pipeline(data_source='directory', data_dir=DATA_DIR)
```

Set `QUICK_DEMO = True` for a faster test run with fewer images and epochs.

## Run Locally

```bash
pip install -r requirements.txt
```

**Option A — CSV file:**
```bash
python facial_expression_recognition.py --csv ./fer2013/fer2013.csv
```

**Option B — image folders:**
```bash
python facial_expression_recognition.py --data-dir ./fer2013/
```

**Quick demo:**
```bash
python facial_expression_recognition.py --demo --data-dir ./fer2013/
```

## Models

### Custom CNN
- 3 convolution blocks (32, 64, 128 filters)
- BatchNorm, MaxPooling, Dropout after each block
- Dense(512) classifier with Softmax output
- Trained on grayscale 48×48 input

### VGG16 (Transfer Learning)
- ImageNet pre-trained weights
- Grayscale replicated to 3 channels
- Frozen base + fine-tuned top layers

### ResNet50 (Transfer Learning)
- Same setup as VGG16
- Fine-tuned top 10 layers

## Evaluation Outputs

The pipeline generates:
- Training/validation accuracy and loss curves
- Normalized confusion matrices
- ROC curves (one-vs-rest AUC)
- Per-class classification reports
- Grad-CAM heatmaps
- Model comparison table and chart
- Misclassified sample analysis

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

## References

- Goodfellow et al. (2013). FER2013 Dataset.
- Simonyan & Zisserman (2014). VGG Networks.
- He et al. (2016). ResNet.
- Selvaraju et al. (2017). Grad-CAM.

---

## GitHub Repository Contents

Push these files to a **public** GitHub repository:

| File / Folder | Required |
|---------------|----------|
| `proposal.md` | Yes — full proposal |
| `facial_expression_recognition.py` | Yes — main code |
| `facial_expression_recognition.ipynb` | Yes — Kaggle notebook |
| `requirements.txt` | Yes |
| `README.md` | Yes |
| `outputs/` | Yes — figures and reports (extract from `outputs.zip`) |

Do **not** upload the dataset or large model files. Link to Kaggle for the dataset instead.

### Before pushing to GitHub

1. Extract `outputs.zip` into the `outputs/` folder (all PNG files + gradcam_cnn/)
2. Create a public repo on GitHub: [Phase-2---Facial-Expressions-Recognition](https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition)
3. Upload all files listed above

```bash
git init
git add proposal.md README.md requirements.txt facial_expression_recognition.py facial_expression_recognition.ipynb outputs/
git commit -m "Phase 2: Facial Expression Recognition proposal and implementation"
git branch -M main
git remote add origin https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition.git
git push -u origin main
```

---

## Teams Submission

**Title:**
```
Phase 2 - Proposal and Code Implementation - Your Name
```

Replace `Your Name` with your actual name.

**Attach or paste in the submission text:**

1. **Proposal** — attach `proposal.md` as PDF, or link to it on GitHub  
2. **GitHub** — https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition  
3. **Kaggle Notebook** — your saved notebook link (Share → Copy link)

**Example submission message:**

```
Phase 2 - Proposal and Code Implementation - [Your Name]

Proposal: [GitHub link to proposal.md or attached PDF]

GitHub Repository: https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition

Kaggle Notebook: https://www.kaggle.com/code/YOUR_USERNAME/your-notebook-name

Best Model: Custom CNN — 64.47% accuracy on FER2013 test set
Dataset: https://www.kaggle.com/datasets/msambare/fer2013
```

### Submission checklist

- [ ] Proposal document complete (`proposal.md`)
- [ ] Public GitHub repo with code, README, and outputs
- [ ] Kaggle notebook saved and link copied
- [ ] README includes dataset link and run instructions
- [ ] Output figures in `outputs/` folder on GitHub
- [ ] Teams title format correct
