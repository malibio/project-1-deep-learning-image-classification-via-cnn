"""
CNN Utilities Module for CIFAR-10 Experiments

This module provides a consistent pipeline for CNN experiments while allowing
notebooks to customize specific components (model architecture, augmentation, etc.).

Usage:
    import cnn_utils
    
    # Load data (consistent across all notebooks)
    data = cnn_utils.load_cifar10_from_tar('cifar-10-python.tar')
    
    # Create model (notebook-specific implementation)
    model = create_model()  # Define this in your notebook
    
    # Train with standard pipeline
    history = cnn_utils.train_model(model, data)
    
    # Evaluate with consistent methods
    cnn_utils.evaluate_model(model, data, history)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
import os
import time
import tarfile
import pickle
import requests
import gzip


# =============================================================================
# FIXED PIPELINE FUNCTIONS (Consistent across all notebooks)
# =============================================================================

def download_cifar10():
    """
    Download CIFAR-10 dataset if it doesn't exist locally.
    
    Returns:
        str: Path to the downloaded tar file
    """
    working_path = os.getcwd()
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name_gz = "cifar-10-python.tar.gz"
    file_name_tar = "cifar-10-python.tar"

    file_path_gz = os.path.join(working_path, file_name_gz)
    file_path_tar = os.path.join(working_path, file_name_tar)

    if os.path.exists(file_path_tar):
        print("The file cifar-10-python.tar already exists")
    else:
        print("Downloading CIFAR-10 data...")
        response = requests.get(cifar_url, stream=True)
        response.raise_for_status()
        with open(file_path_gz, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with gzip.open(file_path_gz, 'rb') as f_in:
            with open(file_path_tar, 'wb') as f_out:
                for chunk in f_in:
                    f_out.write(chunk)
        print("Download completed!")
        
        # Clean up the .gz file
        if os.path.exists(file_path_gz):
            os.remove(file_path_gz)
            
    return file_path_tar


def load_cifar10_from_tar(tar_path=None, auto_download=True):
    """
    Load raw CIFAR-10 data from tar file (without preprocessing).
    
    Args:
        tar_path (str): Path to CIFAR-10 tar file (if None, uses default)
        auto_download (bool): Whether to download if file doesn't exist
        
    Returns:
        dict: Contains raw_train_data, train_labels, raw_test_data, test_labels, label_names
    """
    # Handle tar file path
    if tar_path is None:
        tar_path = 'cifar-10-python.tar'
    
    # Download if file doesn't exist and auto_download is True
    if not os.path.exists(tar_path) and auto_download:
        print(f"CIFAR-10 tar file not found at {tar_path}")
        tar_path = download_cifar10()
    elif not os.path.exists(tar_path):
        raise FileNotFoundError(f"CIFAR-10 tar file not found at {tar_path}")
    
    def load_pickle_from_tar(tar_path, pickle_path):
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(pickle_path)
            f = tar.extractfile(member)
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict

    print("Loading CIFAR-10 data...")
    
    # Load training data
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_path = f'cifar-10-batches-py/data_batch_{i}'
        print(f"Loading training batch {i}...")
        batch_dict = load_pickle_from_tar(tar_path, batch_path)
        train_data.append(batch_dict[b'data'])
        train_labels.extend(batch_dict[b'labels'])
    
    train_data = np.vstack(train_data)
    
    # Load test data
    print("Loading test data...")
    test_batch_path = 'cifar-10-batches-py/test_batch'
    test_dict = load_pickle_from_tar(tar_path, test_batch_path)
    test_data = test_dict[b'data']
    test_labels = test_dict[b'labels']
    
    # Load label names
    meta_path = 'cifar-10-batches-py/batches.meta'
    meta_dict = load_pickle_from_tar(tar_path, meta_path)
    label_names = [label.decode('utf-8') for label in meta_dict[b'label_names']]
    
    print(f"Raw data loaded successfully!")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Label names: {label_names}")
    
    return {
        'raw_train_data': train_data,
        'train_labels': train_labels,
        'raw_test_data': test_data, 
        'test_labels': test_labels,
        'label_names': label_names
    }


def preprocess_data(data_dict, val_split=0.1):
    """
    Preprocess raw CIFAR-10 data: reshape, normalize, one-hot encode, create validation split.
    
    Args:
        data_dict: Dictionary from load_cifar10_from_tar() containing raw data
        val_split: Fraction of training data for validation (default: 0.1)
        
    Returns:
        dict: Contains X_train, y_train, X_val, y_val, X_test, y_test, label_names
    """
    print("Preprocessing data...")
    
    # Extract raw data from dictionary
    train_data = data_dict['raw_train_data']
    train_labels = data_dict['train_labels']
    test_data = data_dict['raw_test_data']
    test_labels = data_dict['test_labels']
    label_names = data_dict['label_names']
    
    # Reshape to image format (N, 32, 32, 3)
    X_train = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(np.array(train_labels), 10)
    y_test = to_categorical(np.array(test_labels), 10)
    
    # Create validation split
    val_size = int(val_split * X_train.shape[0])
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"Preprocessing completed!")
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'label_names': label_names
    }


def train_model(model, data, augmentation=None, epochs=50, batch_size=64, callbacks=None):
    """
    Train a CNN model with optional data augmentation.
    
    Args:
        model: Compiled Keras model
        data: Data dictionary from load_cifar10_from_tar()
        augmentation: Data augmentation generator (optional)
        epochs: Number of training epochs
        batch_size: Training batch size
        callbacks: List of Keras callbacks (optional)
        
    Returns:
        History object from model.fit()
    """
    if callbacks is None:
        callbacks = get_default_callbacks()
    
    print("Starting model training...")
    start_time = time.time()
    
    if augmentation is not None:
        # Train with data augmentation
        history = model.fit(
            augmentation.flow(data['X_train'], data['y_train'], batch_size=batch_size),
            epochs=epochs,
            validation_data=(data['X_val'], data['y_val']),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Train without augmentation
        history = model.fit(
            data['X_train'], data['y_train'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(data['X_val'], data['y_val']),
            callbacks=callbacks,
            verbose=1
        )
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return history


def evaluate_model(model, data, history, save_model_path=None):
    """
    Complete evaluation suite: plots, metrics, confusion matrix, predictions.
    
    Args:
        model: Trained Keras model
        data: Data dictionary from load_cifar10_from_tar()
        history: Training history from train_model()
        save_model_path: Optional path to save the model
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\n🎯 Final Test Results:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = model.predict(data['X_test'], verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(data['y_test'], axis=1)
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=data['label_names']))
    
    # Confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, data['label_names'])
    
    # Visualize predictions
    visualize_predictions(data['X_test'], y_true_classes, y_pred_classes, 
                         data['label_names'])
    
    # Save model if requested
    if save_model_path:
        save_model(model, save_model_path)


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def visualize_predictions(X_test, y_true, y_pred, class_names, num_images=25):
    """Visualize sample predictions with true/predicted labels."""
    sample_indices = np.random.choice(len(X_test), num_images, replace=False)
    
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        idx = sample_indices[i]
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[idx])
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        color = 'green' if true_label == pred_label else 'red'
        
        plt.xlabel(f"T: {true_label}\nP: {pred_label}", color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_data_samples(data, num_images=25):
    """Visualize sample images from the dataset."""
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data['X_train'][i])
        plt.xlabel(data['label_names'][np.argmax(data['y_train'][i])])
    plt.tight_layout()
    plt.show()


def save_model(model, filepath):
    """Save trained model to file."""
    model.save(filepath)
    print(f"Model saved to: {filepath}")


def get_default_callbacks(patience=15):
    """Get default training callbacks."""
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    ]


# =============================================================================
# PLUGGABLE FUNCTIONS (Override in notebooks for custom behavior)
# =============================================================================

def get_training_config():
    """
    Default training configuration.
    Override this function in your notebook for custom training settings.
    """
    return {
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_model_summary(model):
    """Print formatted model summary."""
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📊 Total Parameters: {total_params:,}")


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Example usage docstring
"""
EXAMPLE NOTEBOOK USAGE:

```python
import cnn_utils
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Load raw data (utility handles download if needed)
data_dict = cnn_utils.load_cifar10_from_tar()  # Auto-downloads if not found

# Preprocess data (visible step in notebook)
data = cnn_utils.preprocess_data(data_dict)

# Visualize samples (handled by utility)
cnn_utils.visualize_data_samples(data)

# Define YOUR custom model architecture (notebook-specific)
def create_my_residual_model():
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    # ... your custom residual blocks here ...
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define YOUR custom augmentation strategy (notebook-specific)
def create_my_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

# Create YOUR model and augmentation (notebook provides these)
model = create_my_residual_model()
cnn_utils.print_model_summary(model)

# Optional: create custom augmentation, or pass None for no augmentation
augmentation = create_my_augmentation()  # or None
if augmentation:
    augmentation.fit(data['X_train'])

# Training and evaluation pipeline (handled by utility)
history = cnn_utils.train_model(model, data, augmentation=augmentation)
cnn_utils.evaluate_model(model, data, history, save_model_path='my_experiment.keras')
```

DIFFERENT NOTEBOOK EXAMPLES:

# experiment_simple.ipynb - No augmentation
```python
import cnn_utils
data_dict = cnn_utils.load_cifar10_from_tar()
data = cnn_utils.preprocess_data(data_dict)

model = create_simple_cnn()  # Your basic model
history = cnn_utils.train_model(model, data, augmentation=None)  # No augmentation
cnn_utils.evaluate_model(model, data, history)
```

# experiment_custom_preprocessing.ipynb - Custom preprocessing
```python
import cnn_utils
data_dict = cnn_utils.load_cifar10_from_tar()

# Custom preprocessing with different validation split
data = cnn_utils.preprocess_data(data_dict, val_split=0.15)  # 15% validation

model = create_deep_cnn()  # Your deep model
history = cnn_utils.train_model(model, data)
cnn_utils.evaluate_model(model, data, history)
```

KEY BENEFITS:
1. Clear separation: load → preprocess → train → evaluate
2. Notebook controls preprocessing parameters (validation split, etc.)
3. Visible preprocessing step for debugging/customization
4. Still maintains consistency across experiments
```
"""