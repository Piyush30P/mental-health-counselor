"""
Model Testing Script for Emotion Recognition
Evaluates the trained model on the test dataset.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string
import re

print("="*80)
print("EMOTION RECOGNITION - MODEL TESTING")
print("="*80)

# Define paths
BASE_DIR = r"D:\manoj-major"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

print(f"\nDirectories:")
print(f"  Raw data: {RAW_DATA_DIR}")
print(f"  Preprocessed: {PREPROCESSED_DIR}")
print(f"  Models: {MODELS_DIR}")
print(f"  Outputs: {OUTPUTS_DIR}")

print("\n" + "="*80)

# Load configuration and vocabulary
print("\nLOADING CONFIGURATION:")
print("-" * 80)

try:
    with open(os.path.join(PREPROCESSED_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    print("✓ Configuration loaded")
    
    with open(os.path.join(PREPROCESSED_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    print(f"✓ Vocabulary loaded: {len(word2idx):,} words")
    
    with open(os.path.join(PREPROCESSED_DIR, 'idx2word.pkl'), 'rb') as f:
        idx2word = pickle.load(f)
    print("✓ Index to word mapping loaded")
    
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    sys.exit(1)

emotion_labels = config['emotion_labels']
emotion_names = [emotion_labels[str(i)] for i in range(config['num_classes'])]

print(f"\nConfiguration:")
print(f"  Vocabulary size: {config['vocab_size']:,}")
print(f"  Max sequence length: {config['max_sequence_length']}")
print(f"  Number of classes: {config['num_classes']}")
print(f"  Emotions: {', '.join(emotion_names)}")

print("\n" + "="*80)

# Load test data
print("\nLOADING TEST DATA:")
print("-" * 80)

try:
    df_test = pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"))
    print(f"✓ Test data loaded: {df_test.shape}")
    print(f"  Total samples: {len(df_test):,}")
except Exception as e:
    print(f"✗ Error loading test data: {e}")
    sys.exit(1)

# Display label distribution
print(f"\nTest Label Distribution:")
test_label_counts = df_test['label'].value_counts().sort_index()
for label, count in test_label_counts.items():
    percentage = (count / len(df_test)) * 100
    print(f"  {label} ({emotion_names[label]:10s}): {count:5,} samples ({percentage:5.2f}%)")

print("\n" + "="*80)

# Preprocessing functions
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "can't've": "cannot have", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "I would", "i'll": "I will", "i'm": "I am", "i've": "I have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

def clean_text(text):
    """Clean and standardize text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_punctuation(tokens):
    """Remove punctuation from tokens."""
    return [token for token in tokens if token not in string.punctuation]

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, word2idx, max_length):
    """Preprocess text and convert to sequence."""
    # Clean
    text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = remove_punctuation(tokens)
    
    # Lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    
    # Convert to indices
    indices = [word2idx.get(token, 0) for token in lemmatized]
    
    # Pad/truncate
    if len(indices) < max_length:
        indices = indices + [0] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return indices

# Preprocess test data
print("\nPREPROCESSING TEST DATA:")
print("-" * 80)

X_test_processed = []
for i, text in enumerate(df_test['text']):
    if i % 1000 == 0:
        print(f"  Processing: {i}/{len(df_test)}", end='\r')
    processed = preprocess_text(text, word2idx, config['max_sequence_length'])
    X_test_processed.append(processed)

print(f"  Processing: {len(df_test)}/{len(df_test)}")

X_test = np.array(X_test_processed)
y_test = df_test['label'].values
y_test_categorical = to_categorical(y_test, num_classes=config['num_classes'])

print(f"\n✓ Test data preprocessed:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")

print("\n" + "="*80)

# Load trained model
print("\nLOADING TRAINED MODEL:")
print("-" * 80)

try:
    model = load_model(os.path.join(MODELS_DIR, 'best_emotion_model.h5'))
    print("✓ Best model loaded successfully")
except Exception as e:
    print(f"⚠ Could not load best model: {e}")
    try:
        model = load_model(os.path.join(MODELS_DIR, 'emotion_recognition_cnn_lstm_final.h5'))
        print("✓ Final model loaded successfully")
    except Exception as e2:
        print(f"✗ Error loading model: {e2}")
        sys.exit(1)

print("\nModel Summary:")
model.summary()

print("\n" + "="*80)

# Evaluate model
print("\nEVALUATING MODEL ON TEST SET:")
print("-" * 80)

# Calculate metrics
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
y_test_pred_probs = model.predict(X_test, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
y_test_true_classes = y_test

# Additional metrics
precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')
f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')

print(f"\n📈 ADDITIONAL METRICS:")
print(f"  Precision (weighted): {precision:.4f}")
print(f"  Recall (weighted):    {recall:.4f}")
print(f"  F1-Score (weighted):  {f1:.4f}")

print("\n" + "="*80)

# Classification report
print("\nCLASSIFICATION REPORT:")
print("-" * 80)
print(classification_report(y_test_true_classes, y_test_pred_classes, target_names=emotion_names))

print("="*80)

# Confusion matrix
print("\nGENERATING CONFUSION MATRIX:")
print("-" * 80)

cm_test = confusion_matrix(y_test_true_classes, y_test_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_test, annot=True, fmt='d', cmap='Greens',
    xticklabels=emotion_names,
    yticklabels=emotion_names,
    annot_kws={'size': 12}
)
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Emotion', fontsize=13, fontweight='bold')
plt.ylabel('True Emotion', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Confusion matrix saved: {os.path.join(OUTPUTS_DIR, 'test_confusion_matrix.png')}")

# Per-class accuracy
print(f"\nPER-CLASS ACCURACY:")
print("-" * 80)
per_class_acc = {}
for i, emotion in enumerate(emotion_names):
    class_correct = cm_test[i, i]
    class_total = cm_test[i, :].sum()
    class_acc = class_correct / class_total if class_total > 0 else 0
    per_class_acc[emotion] = float(class_acc)
    print(f"  {emotion:12s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_correct}/{class_total}")

print("\n" + "="*80)

# Save test results
print("\nSAVING TEST RESULTS:")
print("-" * 80)

# Create results dataframe
test_results_df = df_test.copy()
test_results_df['predicted_label'] = y_test_pred_classes
test_results_df['predicted_emotion'] = [emotion_names[i] for i in y_test_pred_classes]
test_results_df['true_emotion'] = [emotion_names[i] for i in y_test_true_classes]
test_results_df['confidence'] = [y_test_pred_probs[i][y_test_pred_classes[i]] * 100 
                                  for i in range(len(y_test_pred_classes))]
test_results_df['correct'] = y_test_true_classes == y_test_pred_classes

# Add probability scores
for i, emotion in enumerate(emotion_names):
    test_results_df[f'prob_{emotion}'] = y_test_pred_probs[:, i] * 100

# Save predictions
test_results_df.to_csv(os.path.join(OUTPUTS_DIR, 'test_predictions.csv'), index=False)
print(f"✓ Test predictions saved: {os.path.join(OUTPUTS_DIR, 'test_predictions.csv')}")

# Save metrics
test_metrics = {
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm_test.tolist(),
    'per_class_accuracy': per_class_acc
}

with open(os.path.join(OUTPUTS_DIR, 'test_metrics.json'), 'w') as f:
    json.dump(test_metrics, f, indent=4)
print(f"✓ Test metrics saved: {os.path.join(OUTPUTS_DIR, 'test_metrics.json')}")

print("\n" + "="*80)

# Display sample predictions
print("\nSAMPLE PREDICTIONS:")
print("-" * 80)

np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(sample_indices):
    text = df_test.iloc[idx]['text']
    true_label = y_test_true_classes[idx]
    pred_label = y_test_pred_classes[idx]
    confidence = y_test_pred_probs[idx][pred_label] * 100
    
    print(f"\nSample {i+1}:")
    print(f"  Text: {text}")
    print(f"  True: {emotion_names[true_label]}")
    print(f"  Predicted: {emotion_names[pred_label]} ({confidence:.2f}%)")
    print(f"  Status: {'✓ CORRECT' if true_label == pred_label else '✗ INCORRECT'}")

print("\n" + "="*80)
print("✨ TESTING COMPLETED SUCCESSFULLY!")
print("="*80)

# Final summary
print(f"\nFINAL SUMMARY:")
print(f"  Total test samples: {len(test_results_df):,}")
print(f"  Correct predictions: {test_results_df['correct'].sum():,} ({test_results_df['correct'].sum()/len(test_results_df)*100:.2f}%)")
print(f"  Average confidence: {test_results_df['confidence'].mean():.2f}%")
print(f"\n  All results saved to: {OUTPUTS_DIR}")
