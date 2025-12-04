"""
Model Training Script for Emotion Recognition
Loads preprocessed data and trains the CNN-LSTM model.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, SpatialDropout1D,
    BatchNormalization, LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print("="*80)
print("EMOTION RECOGNITION - MODEL TRAINING")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Define paths
BASE_DIR = r"D:\manoj-major"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"\nDirectories:")
print(f"  Preprocessed data: {PREPROCESSED_DIR}")
print(f"  Models output: {MODELS_DIR}")
print(f"  Training outputs: {OUTPUTS_DIR}")
print("\n" + "="*80)

# Load preprocessed data
print("\nLOADING PREPROCESSED DATA:")
print("-" * 80)

try:
    X = np.load(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'))
    print(f"✓ Loaded sequences: X shape = {X.shape}")
    
    y = np.load(os.path.join(PREPROCESSED_DIR, 'y_labels.npy'))
    print(f"✓ Loaded labels: y shape = {y.shape}")
    
    embedding_matrix = np.load(os.path.join(PREPROCESSED_DIR, 'embedding_matrix.npy'))
    print(f"✓ Loaded embeddings: shape = {embedding_matrix.shape}")
    
    with open(os.path.join(PREPROCESSED_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    print(f"✓ Loaded vocabulary: {len(word2idx):,} words")
    
    with open(os.path.join(PREPROCESSED_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    print(f"✓ Loaded configuration")
    
except Exception as e:
    print(f"✗ Error loading preprocessed data: {e}")
    print("\nPlease run the preprocessing script first: python scripts/1_preprocess_data.py")
    sys.exit(1)

print(f"\nConfiguration:")
print(f"  Vocabulary size: {config['vocab_size']:,}")
print(f"  Embedding dimension: {config['embedding_dim']}")
print(f"  Max sequence length: {config['max_sequence_length']}")
print(f"  Number of classes: {config['num_classes']}")

print("\n" + "="*80)

# Prepare data
print("\nPREPARING DATA:")
print("-" * 80)

# Convert labels to categorical
y_categorical = to_categorical(y, num_classes=config['num_classes'])
print(f"Categorical labels shape: {y_categorical.shape}")

# Display label distribution
emotion_labels = config['emotion_labels']
print(f"\nLabel Distribution:")
for label_idx in range(config['num_classes']):
    count = np.sum(y == label_idx)
    percentage = (count / len(y)) * 100
    print(f"  {label_idx} ({emotion_labels[str(label_idx)]:10s}): {count:5,} samples ({percentage:5.2f}%)")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain-Validation Split:")
print(f"  Training set:   X={X_train.shape}, y={y_train.shape}")
print(f"  Validation set: X={X_val.shape}, y={y_val.shape}")

print("\n" + "="*80)

# Build model
def build_cnn_lstm_model(vocab_size, embedding_dim, max_length, num_classes, embedding_matrix):
    """Build CNN-LSTM model for emotion recognition."""
    
    print("\nBUILDING CNN-LSTM MODEL:")
    print("-" * 80)
    
    model = Sequential([
        # Embedding layer
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=True,
            name='word2vec_embedding'
        ),
        
        # CNN Block 1
        Conv1D(128, 5, activation='relu', padding='same', name='conv1d_block1'),
        MaxPooling1D(2, name='maxpool_block1'),
        SpatialDropout1D(0.2, name='spatial_dropout_block1'),
        BatchNormalization(name='batch_norm_block1'),
        
        # CNN Block 2
        Conv1D(256, 5, activation='relu', padding='same', name='conv1d_block2'),
        MaxPooling1D(2, name='maxpool_block2'),
        SpatialDropout1D(0.2, name='spatial_dropout_block2'),
        BatchNormalization(name='batch_norm_block2'),
        
        # CNN Block 3
        Conv1D(512, 5, activation='relu', padding='same', name='conv1d_block3'),
        MaxPooling1D(2, name='maxpool_block3'),
        SpatialDropout1D(0.2, name='spatial_dropout_block3'),
        BatchNormalization(name='batch_norm_block3'),
        
        # LSTM layers
        LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm_layer1'),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm_layer2'),
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm_layer3'),
        
        # Classification layers
        Dense(128, activation='relu', name='dense_layer'),
        Dropout(0.5, name='dropout_final'),
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    return model

# Build the model
model = build_cnn_lstm_model(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    max_length=config['max_sequence_length'],
    num_classes=config['num_classes'],
    embedding_matrix=embedding_matrix
)

print("\nModel Summary:")
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])

print(f"\nTotal parameters: {trainable_params + non_trainable_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")

print("\n" + "="*80)

# Compile model
print("\nCOMPILING MODEL:")
print("-" * 80)

optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss: Categorical Crossentropy")
print(f"  Metrics: Accuracy")

print("\n" + "="*80)

# Setup callbacks
print("\nSETTING UP CALLBACKS:")
print("-" * 80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_emotion_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

print("✓ Callbacks configured:")
print("  1. EarlyStopping (patience=5)")
print("  2. ModelCheckpoint (save_best_only=True)")
print("  3. ReduceLROnPlateau (factor=0.5)")

print("\n" + "="*80)

# Train model
print("\nTRAINING MODEL:")
print("-" * 80)
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Batch size: 32")
print(f"Epochs: 50 (with early stopping)")
print("\nStarting training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*80)
print("✓ Training completed!")
print("="*80)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Training plots saved: {os.path.join(OUTPUTS_DIR, 'training_history.png')}")

# Evaluate on validation set
print("\n" + "="*80)
print("VALIDATION SET EVALUATION:")
print("-" * 80)

y_pred_probs = model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

val_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

# Classification report
emotion_names = [emotion_labels[str(i)] for i in range(config['num_classes'])]
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=emotion_names))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=emotion_names,
    yticklabels=emotion_names
)
plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'validation_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix saved: {os.path.join(OUTPUTS_DIR, 'validation_confusion_matrix.png')}")

print("\n" + "="*80)

# Save final model
model.save(os.path.join(MODELS_DIR, 'emotion_recognition_cnn_lstm_final.h5'))
print(f"✓ Final model saved: {os.path.join(MODELS_DIR, 'emotion_recognition_cnn_lstm_final.h5')}")

# Save training history
with open(os.path.join(OUTPUTS_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
print(f"✓ Training history saved: {os.path.join(OUTPUTS_DIR, 'training_history.pkl')}")

# Save model architecture
model_json = model.to_json()
with open(os.path.join(OUTPUTS_DIR, 'model_architecture.json'), 'w') as f:
    f.write(model_json)
print(f"✓ Model architecture saved: {os.path.join(OUTPUTS_DIR, 'model_architecture.json')}")

print("\n" + "="*80)
print("✨ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nNext step: Run the testing script (3_test_model.py)")
