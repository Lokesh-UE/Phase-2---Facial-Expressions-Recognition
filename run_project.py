"""
Facial Expression Recognition 
Author: Lokesh Chamakuri
Phase 2: Proposal and Code Implementation

This is the ONE file needed to run the entire project.
Install dependencies first: pip install -r requirements.txt

Dataset (download separately):
  https://www.kaggle.com/datasets/msambare/fer2013

Run locally:
  python run_project.py --data-dir ./fer2013/
  python run_project.py --demo --data-dir ./fer2013/

Run on Kaggle:
  Add msambare/fer2013 dataset, enable GPU, then:
  python run_project.py

GitHub:
  https://github.com/Lokesh-UE/Phase-2---Facial-Expressions-Recognition

PROJECT FOLDER REFERENCE
==========================
facial_expression_recognition/
|
|-- run_project.py                          <- THIS FILE (all code, run this)
|-- facial_expression_recognition.py        <- same code (alternate name)
|-- facial_expression_recognition.ipynb     <- Kaggle notebook version
|-- requirements.txt                        <- pip install -r requirements.txt
|-- README.md                               <- setup and run instructions
|-- proposal.md                             <- full project proposal (markdown)
|-- Phase 2 - Proposal and Code Implementation - Lokesh Chamakuri.docx
|-- .gitignore                              <- excludes dataset and checkpoints
|
|-- outputs/                                <- generated after training
|   |-- 01_sample_images.png
|   |-- 02_class_distribution.png
|   |-- 03_cnn_training_history.png
|   |-- 04_cnn_confusion_matrix.png
|   |-- 05_cnn_roc_curves.png
|   |-- 06_vgg_training_history.png
|   |-- 06_vgg_confusion_matrix.png
|   |-- 07_vgg_roc_curves.png
|   |-- 08_resnet_training_history.png
|   |-- 08_resnet_confusion_matrix.png
|   |-- 09_resnet_roc_curves.png
|   |-- 10_model_comparison.png
|   |-- 11_error_analysis.png
|   |-- model_comparison.csv
|   |-- custom_cnn_classification_report.txt
|   |-- vgg16_classification_report.txt
|   |-- resnet50_classification_report.txt
|   |-- gradcam_cnn/                        <- Grad-CAM heatmap images
|
|-- checkpoints/                            <- saved models (after training)
|   |-- custom_cnn_final.keras
|   |-- vgg16_final.keras
|   |-- resnet50_final.keras
|
|-- fer2013/                                <- download dataset here (not in GitHub)
    |-- train/angry/, train/happy/, ...
    |-- test/angry/, test/happy/, ...
    OR fer2013.csv

RESULTS (best model: Custom CNN — 64.47% accuracy)
  Custom CNN  64.47%  |  VGG16  49.51%  |  ResNet50  32.81%
"""

PROJECT_FILES = {
    "run_project.py": "Main script — run the full pipeline (this file)",
    "facial_expression_recognition.py": "Same code as run_project.py",
    "facial_expression_recognition.ipynb": "Jupyter/Kaggle notebook version",
    "requirements.txt": "Python packages: tensorflow, numpy, pandas, matplotlib, seaborn, scikit-learn, opencv-python",
    "README.md": "Project overview, results, and run instructions",
    "proposal.md": "Complete Phase 2 proposal document",
    "Phase 2 - Proposal and Code Implementation - Lokesh Chamakuri.docx": "Proposal in Word format",
    "outputs/": "Training figures, CSV comparison table, classification reports",
    "checkpoints/": "Saved Keras models after training",
    "fer2013/": "FER2013 dataset folder (download from Kaggle, not stored on GitHub)",
}


def show_project_files():
    print("\nProject files in this repository:\n")
    for path, description in PROJECT_FILES.items():
        print(f"  {path:<60} {description}")
    print("\nDataset link: https://www.kaggle.com/datasets/msambare/fer2013\n")


import os
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


class Config:
    DATA_PATH = '/kaggle/input/fer2013/'
    IMG_SIZE = 48
    CHANNELS = 1
    NUM_CLASSES = 7
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    ROTATION_RANGE = 15
    WIDTH_SHIFT = 0.1
    HEIGHT_SHIFT = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    CHECKPOINT_DIR = './checkpoints/'
    OUTPUT_DIR = './outputs/'
    RANDOM_SEED = 42
    QUICK_DEMO = False
    DEMO_SAMPLES_PER_CLASS = 200
    DEMO_EPOCHS = 3


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

COLORS = ['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#95a5a6']
CSV_NAMES = ['fer2013.csv', 'icml_face_data.csv', 'train.csv']
KNOWN_DATA_DIRS = [
    '/kaggle/input/datasets/msambare/fer2013',
    '/kaggle/input/fer2013',
    './fer2013',
]
KNOWN_CSV_PATHS = [
    '/kaggle/input/datasets/msambare/fer2013/fer2013.csv',
    '/kaggle/input/fer2013/fer2013.csv',
    './fer2013/fer2013.csv',
]


def find_fer2013_data(search_roots=None):
    for csv_path in KNOWN_CSV_PATHS:
        if os.path.exists(csv_path):
            return csv_path, None
    for data_dir in KNOWN_DATA_DIRS:
        train_path = os.path.join(data_dir, 'train')
        if os.path.isdir(train_path):
            return None, data_dir
    search_roots = search_roots or ['/kaggle/input', './fer2013', '.']
    csv_path = None
    data_dir = None
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            lower_names = {name.lower(): name for name in filenames}
            for csv_name in CSV_NAMES:
                if csv_name in lower_names:
                    candidate = os.path.join(dirpath, lower_names[csv_name])
                    if csv_name == 'train.csv' and 'test.csv' not in lower_names:
                        continue
                    csv_path = candidate
                    break
            if csv_path:
                break
            if 'train' in dirnames:
                train_path = os.path.join(dirpath, 'train')
                if os.path.isdir(train_path):
                    subdirs = [d.lower() for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
                    if any(emotion.lower() in subdirs for emotion in Config.EMOTIONS):
                        data_dir = dirpath
            if data_dir:
                break
        if csv_path or data_dir:
            break
    return csv_path, data_dir


def load_fer2013_csv(csv_path='fer2013.csv'):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(df['emotion'].value_counts().sort_index())
    return df


def parse_pixels(pixel_string):
    return np.array(pixel_string.split(), dtype='float32')


def prepare_data(df):
    df['pixels'] = df['pixels'].apply(parse_pixels)
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']
    X_train = np.stack(train_df['pixels'].values)
    y_train = train_df['emotion'].values
    X_val = np.stack(val_df['pixels'].values)
    y_val = val_df['emotion'].values
    X_test = np.stack(test_df['pixels'].values)
    y_test = test_df['emotion'].values
    X_train = X_train.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE, 1)
    X_val = X_val.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE, 1)
    X_test = X_test.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE, 1)
    print(f"Training: {X_train.shape[0]}")
    print(f"Validation: {X_val.shape[0]}")
    print(f"Test: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_from_directory(data_dir, quick_demo=False):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    train_limit = Config.DEMO_SAMPLES_PER_CLASS if quick_demo else None
    test_limit = 50 if quick_demo else None

    def load_split(split_name, max_per_class=None):
        X, y = [], []
        split_path = os.path.join(data_dir, split_name)
        print(f"Loading {split_name} images...")
        for emotion_idx, emotion in enumerate(Config.EMOTIONS):
            emotion_path = os.path.join(split_path, emotion.lower())
            if not os.path.exists(emotion_path):
                continue
            img_names = sorted(os.listdir(emotion_path))
            if max_per_class and len(img_names) > max_per_class:
                rng = np.random.RandomState(Config.RANDOM_SEED + emotion_idx)
                img_names = list(rng.choice(img_names, max_per_class, replace=False))
            for img_name in img_names:
                img_path = os.path.join(emotion_path, img_name)
                img = load_img(img_path, color_mode='grayscale', target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
                X.append(img_to_array(img))
                y.append(emotion_idx)
            print(f"  {emotion}: {len(img_names)}")
        return np.array(X), np.array(y)

    X_train, y_train = load_split('train', max_per_class=train_limit)
    X_test, y_test = load_split('test', max_per_class=test_limit)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=Config.VALIDATION_SPLIT,
        random_state=Config.RANDOM_SEED, stratify=y_train
    )
    print(f"Training: {X_train.shape[0]}")
    print(f"Validation: {X_val.shape[0]}")
    print(f"Test: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_data(X_train, X_val, X_test):
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, X_val, X_test


def encode_labels(y_train, y_val, y_test):
    y_train_cat = to_categorical(y_train, Config.NUM_CLASSES)
    y_val_cat = to_categorical(y_val, Config.NUM_CLASSES)
    y_test_cat = to_categorical(y_test, Config.NUM_CLASSES)
    return y_train_cat, y_val_cat, y_test_cat


def compute_class_weights(y_train):
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    for i, w in class_weight_dict.items():
        print(f"{Config.EMOTIONS[i]}: {w:.3f}")
    return class_weight_dict


def create_data_generators(X_train, y_train_cat, X_val, y_val_cat):
    train_datagen = ImageDataGenerator(
        rotation_range=Config.ROTATION_RANGE, width_shift_range=Config.WIDTH_SHIFT,
        height_shift_range=Config.HEIGHT_SHIFT, zoom_range=Config.ZOOM_RANGE,
        horizontal_flip=Config.HORIZONTAL_FLIP, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=Config.BATCH_SIZE, shuffle=True, seed=Config.RANDOM_SEED)
    val_generator = val_datagen.flow(X_val, y_val_cat, batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_generator, val_generator


def subsample_for_demo(X, y, samples_per_class=None):
    samples_per_class = samples_per_class or Config.DEMO_SAMPLES_PER_CLASS
    indices = []
    for class_idx in np.unique(y):
        class_indices = np.where(y == class_idx)[0]
        n = min(samples_per_class, len(class_indices))
        indices.extend(np.random.choice(class_indices, n, replace=False))
    indices = np.array(indices)
    np.random.shuffle(indices)
    return X[indices], y[indices]


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


def merge_histories(*histories):
    merged = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    for history in histories:
        for key in merged:
            merged[key].extend(history.history[key])
    return merged


def plot_sample_images(X, y, save_path=None):
    fig, axes = plt.subplots(1, Config.NUM_CLASSES, figsize=(16, 3))
    for i, emotion in enumerate(Config.EMOTIONS):
        idx = np.where(y == i)[0][0]
        axes[i].imshow(X[idx].squeeze(), cmap='gray')
        axes[i].set_title(emotion)
        axes[i].axis('off')
    plt.suptitle('Sample images from FER2013')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_distribution(y_train, y_val, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    x = np.arange(Config.NUM_CLASSES)
    width = 0.25
    ax.bar(x - width, train_counts, width, label='Training')
    ax.bar(x, val_counts, width, label='Validation')
    ax.bar(x + width, test_counts, width, label='Test')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_title('Class distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.EMOTIONS, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history, model_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title(f'{model_name} accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title(f'{model_name} loss')
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=Config.EMOTIONS, yticklabels=Config.EMOTIONS, ax=ax)
    ax.set_title(f'{model_name} confusion matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true, y_pred_proba, model_name, save_path=None):
    y_true_bin = label_binarize(y_true, classes=range(Config.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, emotion in enumerate(Config.EMOTIONS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{emotion} (AUC={roc_auc:.3f})', color=COLORS[i])
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'{model_name} ROC curves')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
        ax.bar(x + i * width, values, width, label=row['Model'])
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def build_custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(), Conv2D(32, (3, 3), padding='same', activation='relu'), BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'), BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'), BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),
        Conv2D(128, (3, 3), padding='same', activation='relu'), BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'), BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),
        Flatten(), Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_vgg16_transfer(input_shape=(48, 48, 3), num_classes=7):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE / 10), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model


def build_resnet50_transfer(input_shape=(48, 48, 3), num_classes=7):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE / 10), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model


def get_callbacks(model_name):
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=os.path.join(Config.CHECKPOINT_DIR, f'{model_name}_best.keras'),
                        monitor='val_accuracy', save_best_only=True, verbose=1)
    ]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    last_conv_idx = next(i for i, layer in enumerate(model.layers) if layer.name == last_conv_layer_name)
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == last_conv_idx:
                conv_outputs = x
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def visualize_gradcam_samples(model, X_test, y_test, model_name, last_conv_layer_name, num_samples=5, save_dir=None):
    import cv2
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    for i, idx in enumerate(indices):
        try:
            img = X_test[idx]
            img_array = np.expand_dims(img, axis=0)
            pred = model.predict(img_array, verbose=0)
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_path = os.path.join(save_dir, f'gradcam_{model_name}_{i}.png') if save_dir else None
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img.squeeze(), cmap='gray')
            axes[0].set_title(f"True: {Config.EMOTIONS[y_test[idx]]}, Pred: {Config.EMOTIONS[np.argmax(pred)]}")
            axes[0].axis('off')
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            superimposed = plt.cm.jet(heatmap_resized)[:, :, :3] * 0.4 + np.repeat(img, 3, axis=-1) * 0.6
            axes[1].imshow(np.clip(superimposed, 0, 1))
            axes[1].set_title('Grad-CAM')
            axes[1].axis('off')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Grad-CAM skipped for sample {i}: {e}")


def evaluate_model(model, X_test, y_test, y_test_cat, model_name):
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n{model_name} results")
    print(f"Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    report = classification_report(y_test, y_pred, target_names=Config.EMOTIONS, zero_division=0)
    print(report)
    report_path = os.path.join(Config.OUTPUT_DIR, f'{model_name.replace(" ", "_").lower()}_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return {'Model': model_name, 'Test_Loss': test_loss, 'Test_Accuracy': test_acc, 'Accuracy': accuracy,
            'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba}


def run_pipeline(data_source='csv', csv_path=None, data_dir=None):
    print("Starting facial expression recognition pipeline")
    if data_source == 'csv' and csv_path:
        df = load_fer2013_csv(csv_path)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(df)
    elif data_source == 'directory' and data_dir:
        if Config.QUICK_DEMO:
            print("Demo mode: loading small subset of images")
        X_train, y_train, X_val, y_val, X_test, y_test = load_from_directory(data_dir, quick_demo=Config.QUICK_DEMO)
    else:
        raise ValueError("Invalid data source or path.")

    plot_sample_images(X_train, y_train, save_path=os.path.join(Config.OUTPUT_DIR, '01_sample_images.png'))
    plot_class_distribution(y_train, y_val, y_test, save_path=os.path.join(Config.OUTPUT_DIR, '02_class_distribution.png'))
    epochs = Config.DEMO_EPOCHS if Config.QUICK_DEMO else Config.EPOCHS
    if Config.QUICK_DEMO and data_source == 'csv':
        X_train, y_train = subsample_for_demo(X_train, y_train)
        X_val, y_val = subsample_for_demo(X_val, y_val, samples_per_class=50)
        X_test, y_test = subsample_for_demo(X_test, y_test, samples_per_class=50)
    elif Config.QUICK_DEMO:
        print(f"Demo mode: {epochs} epochs per model")

    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    y_train_cat, y_val_cat, y_test_cat = encode_labels(y_train, y_val, y_test)
    class_weights = compute_class_weights(y_train)
    train_generator, val_generator = create_data_generators(X_train, y_train_cat, X_val, y_val_cat)

    custom_cnn = build_custom_cnn(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 1))
    custom_cnn.summary()
    history_cnn = custom_cnn.fit(train_generator, epochs=epochs, validation_data=val_generator,
                                 callbacks=get_callbacks('CustomCNN'), class_weight=class_weights, verbose=1)
    plot_training_history(history_cnn, 'Custom CNN', save_path=os.path.join(Config.OUTPUT_DIR, '03_cnn_training_history.png'))
    results_cnn = evaluate_model(custom_cnn, X_test, y_test, y_test_cat, 'Custom CNN')
    plot_confusion_matrix(y_test, results_cnn['y_pred'], 'Custom CNN', save_path=os.path.join(Config.OUTPUT_DIR, '04_cnn_confusion_matrix.png'))
    plot_roc_curves(y_test, results_cnn['y_pred_proba'], 'Custom CNN', save_path=os.path.join(Config.OUTPUT_DIR, '05_cnn_roc_curves.png'))
    try:
        visualize_gradcam_samples(custom_cnn, X_test, y_test, 'CustomCNN', get_last_conv_layer_name(custom_cnn),
                                  num_samples=5, save_dir=os.path.join(Config.OUTPUT_DIR, 'gradcam_cnn'))
    except Exception as e:
        print(f"Grad-CAM step skipped: {e}")

    X_train_rgb = np.repeat(X_train, 3, axis=-1)
    X_val_rgb = np.repeat(X_val, 3, axis=-1)
    X_test_rgb = np.repeat(X_test, 3, axis=-1)
    train_datagen_rgb = ImageDataGenerator(rotation_range=Config.ROTATION_RANGE, width_shift_range=Config.WIDTH_SHIFT,
        height_shift_range=Config.HEIGHT_SHIFT, zoom_range=Config.ZOOM_RANGE, horizontal_flip=Config.HORIZONTAL_FLIP)
    val_datagen_rgb = ImageDataGenerator()
    train_gen_rgb = train_datagen_rgb.flow(X_train_rgb, y_train_cat, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_gen_rgb = val_datagen_rgb.flow(X_val_rgb, y_val_cat, batch_size=Config.BATCH_SIZE, shuffle=False)

    vgg_model, vgg_base = build_vgg16_transfer(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    phase1_epochs = 2 if Config.QUICK_DEMO else 15
    phase2_epochs = 2 if Config.QUICK_DEMO else 20
    history_vgg_phase1 = vgg_model.fit(train_gen_rgb, epochs=phase1_epochs, validation_data=val_gen_rgb,
        callbacks=get_callbacks('VGG16_Phase1'), class_weight=class_weights, verbose=1)
    for layer in vgg_base.layers[-4:]:
        layer.trainable = True
    vgg_model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE / 100), loss='categorical_crossentropy', metrics=['accuracy'])
    history_vgg_phase2 = vgg_model.fit(train_gen_rgb, epochs=phase2_epochs, validation_data=val_gen_rgb,
        callbacks=get_callbacks('VGG16_FineTuned'), class_weight=class_weights, verbose=1)
    history_vgg = merge_histories(history_vgg_phase1, history_vgg_phase2)
    plot_training_history(type('History', (), {'history': history_vgg})(), 'VGG16', save_path=os.path.join(Config.OUTPUT_DIR, '06_vgg_training_history.png'))
    results_vgg = evaluate_model(vgg_model, X_test_rgb, y_test, y_test_cat, 'VGG16')
    plot_confusion_matrix(y_test, results_vgg['y_pred'], 'VGG16', save_path=os.path.join(Config.OUTPUT_DIR, '06_vgg_confusion_matrix.png'))
    plot_roc_curves(y_test, results_vgg['y_pred_proba'], 'VGG16', save_path=os.path.join(Config.OUTPUT_DIR, '07_vgg_roc_curves.png'))

    resnet_model, resnet_base = build_resnet50_transfer(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    history_resnet_phase1 = resnet_model.fit(train_gen_rgb, epochs=phase1_epochs, validation_data=val_gen_rgb,
        callbacks=get_callbacks('ResNet50_Phase1'), class_weight=class_weights, verbose=1)
    for layer in resnet_base.layers[-10:]:
        layer.trainable = True
    resnet_model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE / 100), loss='categorical_crossentropy', metrics=['accuracy'])
    history_resnet_phase2 = resnet_model.fit(train_gen_rgb, epochs=phase2_epochs, validation_data=val_gen_rgb,
        callbacks=get_callbacks('ResNet50_FineTuned'), class_weight=class_weights, verbose=1)
    history_resnet = merge_histories(history_resnet_phase1, history_resnet_phase2)
    plot_training_history(type('History', (), {'history': history_resnet})(), 'ResNet50', save_path=os.path.join(Config.OUTPUT_DIR, '08_resnet_training_history.png'))
    results_resnet = evaluate_model(resnet_model, X_test_rgb, y_test, y_test_cat, 'ResNet50')
    plot_confusion_matrix(y_test, results_resnet['y_pred'], 'ResNet50', save_path=os.path.join(Config.OUTPUT_DIR, '08_resnet_confusion_matrix.png'))
    plot_roc_curves(y_test, results_resnet['y_pred_proba'], 'ResNet50', save_path=os.path.join(Config.OUTPUT_DIR, '09_resnet_roc_curves.png'))

    comparison_df = pd.DataFrame({
        'Model': ['Custom CNN', 'VGG16', 'ResNet50'],
        'Accuracy': [results_cnn['Accuracy'], results_vgg['Accuracy'], results_resnet['Accuracy']],
        'Precision': [results_cnn['Precision'], results_vgg['Precision'], results_resnet['Precision']],
        'Recall': [results_cnn['Recall'], results_vgg['Recall'], results_resnet['Recall']],
        'F1-Score': [results_cnn['F1-Score'], results_vgg['F1-Score'], results_resnet['F1-Score']],
        'Test_Loss': [results_cnn['Test_Loss'], results_vgg['Test_Loss'], results_resnet['Test_Loss']]
    })
    print(comparison_df)
    comparison_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'model_comparison.csv'), index=False)
    plot_model_comparison(comparison_df, save_path=os.path.join(Config.OUTPUT_DIR, '10_model_comparison.png'))
    custom_cnn.save(os.path.join(Config.CHECKPOINT_DIR, 'custom_cnn_final.keras'))
    vgg_model.save(os.path.join(Config.CHECKPOINT_DIR, 'vgg16_final.keras'))
    resnet_model.save(os.path.join(Config.CHECKPOINT_DIR, 'resnet50_final.keras'))

    misclassified = np.where(results_cnn['y_pred'] != y_test)[0]
    print(f"Misclassified: {len(misclassified)} / {len(y_test)}")
    if len(misclassified) > 0:
        num_errors = min(10, len(misclassified))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for i in range(num_errors):
            idx = misclassified[i]
            axes[i].imshow(X_test[idx].squeeze(), cmap='gray')
            axes[i].set_title(f"True: {Config.EMOTIONS[y_test[idx]]}\nPred: {Config.EMOTIONS[results_cnn['y_pred'][idx]]}", fontsize=9)
            axes[i].axis('off')
        plt.suptitle('Misclassified examples')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, '11_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    print("Done. Outputs in ./outputs/, models in ./checkpoints/")
    return {'custom_cnn': results_cnn, 'vgg16': results_vgg, 'resnet50': results_resnet, 'comparison': comparison_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Expression Recognition — run full project')
    parser.add_argument('--demo', action='store_true', help='Quick test with subset of data')
    parser.add_argument('--csv', type=str, default=None, help='Path to fer2013.csv')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to fer2013 folder with train/test')
    parser.add_argument('--list-files', action='store_true', help='Show all project files and exit')
    args = parser.parse_args()

    if args.list_files:
        show_project_files()
    else:
        if args.demo:
            Config.QUICK_DEMO = True
        if args.csv and os.path.exists(args.csv):
            run_pipeline(data_source='csv', csv_path=args.csv)
        elif args.data_dir and os.path.exists(args.data_dir):
            run_pipeline(data_source='directory', data_dir=args.data_dir)
        else:
            csv_path, data_dir = find_fer2013_data()
            if csv_path:
                run_pipeline(data_source='csv', csv_path=csv_path)
            elif data_dir:
                run_pipeline(data_source='directory', data_dir=data_dir)
            else:
                show_project_files()
                print("Dataset not found.")
                print("Download: https://www.kaggle.com/datasets/msambare/fer2013")
