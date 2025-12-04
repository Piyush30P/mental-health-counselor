"""
Data Preprocessing Script for Emotion Recognition
This script preprocesses raw CSV data and creates all necessary files for training.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Word2Vec
from gensim.models import Word2Vec

# Configuration
print("="*80)
print("EMOTION RECOGNITION - DATA PREPROCESSING")
print("="*80)

# Define paths
BASE_DIR = r"D:\manoj-major"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

# Create directories if they don't exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# Configuration parameters
CONFIG = {
    'max_sequence_length': 100,
    'embedding_dim': 300,
    'min_word_frequency': 2,
    'emotion_labels': {
        '0': 'sadness',
        '1': 'joy',
        '2': 'love',
        '3': 'anger',
        '4': 'fear',
        '5': 'surprise'
    },
    'num_classes': 6
}

print(f"\nConfiguration:")
print(f"  Base Directory: {BASE_DIR}")
print(f"  Raw Data: {RAW_DATA_DIR}")
print(f"  Preprocessed Output: {PREPROCESSED_DIR}")
print(f"  Max Sequence Length: {CONFIG['max_sequence_length']}")
print(f"  Embedding Dimension: {CONFIG['embedding_dim']}")
print("\n" + "="*80)

# Download NLTK data
print("\nDownloading NLTK data...")
print("-" * 80)

nltk_packages = [
    'punkt',
    'stopwords',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]

for package in nltk_packages:
    try:
        nltk.download(package, quiet=True)
        print(f"✓ Downloaded: {package}")
    except Exception as e:
        print(f"⚠ Warning: Could not download {package}: {e}")

print("\n" + "="*80)

# Contraction mapping
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
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Expand contractions
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    # Remove extra whitespace
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

def preprocess_text(text):
    """Complete preprocessing pipeline for text."""
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
    
    # POS tagging and lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    
    return lemmatized

# Load datasets
print("\nLOADING DATASETS:")
print("-" * 80)

try:
    df_train = pd.read_csv(os.path.join(RAW_DATA_DIR, "training.csv"))
    print(f"✓ Training data loaded: {df_train.shape}")
except Exception as e:
    print(f"✗ Error loading training.csv: {e}")
    sys.exit(1)

try:
    df_val = pd.read_csv(os.path.join(RAW_DATA_DIR, "validation.csv"))
    print(f"✓ Validation data loaded: {df_val.shape}")
except Exception as e:
    print(f"✗ Error loading validation.csv: {e}")
    sys.exit(1)

try:
    df_test = pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"))
    print(f"✓ Test data loaded: {df_test.shape}")
except Exception as e:
    print(f"✗ Error loading test.csv: {e}")
    sys.exit(1)

# Combine training and validation for preprocessing
df_combined = pd.concat([df_train, df_val], ignore_index=True)
print(f"\nCombined training dataset: {df_combined.shape}")

# Display label distribution
print(f"\nLabel Distribution (Combined):")
label_counts = df_combined['label'].value_counts().sort_index()
for label, count in label_counts.items():
    percentage = (count / len(df_combined)) * 100
    emotion_name = CONFIG['emotion_labels'][str(label)]
    print(f"  {label} ({emotion_name:10s}): {count:6,} samples ({percentage:5.2f}%)")

print("\n" + "="*80)

# Preprocess texts
print("\nPREPROCESSING TEXTS:")
print("-" * 80)

print("\nProcessing combined dataset...")
processed_texts = []
for text in tqdm(df_combined['text'], desc="Preprocessing"):
    tokens = preprocess_text(text)
    processed_texts.append(tokens)

print(f"✓ Processed {len(processed_texts):,} texts")
print(f"  Average tokens per text: {np.mean([len(t) for t in processed_texts]):.2f}")
print(f"  Max tokens: {max([len(t) for t in processed_texts])}")
print(f"  Min tokens: {min([len(t) for t in processed_texts])}")

print("\n" + "="*80)

# Build vocabulary
print("\nBUILDING VOCABULARY:")
print("-" * 80)

# Count word frequencies
word_freq = Counter()
for tokens in processed_texts:
    word_freq.update(tokens)

print(f"Total unique words (before filtering): {len(word_freq):,}")

# Filter by minimum frequency
filtered_vocab = {word: count for word, count in word_freq.items() 
                  if count >= CONFIG['min_word_frequency']}

print(f"Vocabulary size (min_freq={CONFIG['min_word_frequency']}): {len(filtered_vocab):,}")

# Create word to index mapping (reserve 0 for padding/unknown)
word2idx = {word: idx+1 for idx, word in enumerate(sorted(filtered_vocab.keys()))}
word2idx['<PAD>'] = 0  # Padding token
idx2word = {idx: word for word, idx in word2idx.items()}

CONFIG['vocab_size'] = len(word2idx)

print(f"✓ Vocabulary size (with <PAD>): {CONFIG['vocab_size']:,}")
print(f"\nTop 10 most common words:")
for word, count in word_freq.most_common(10):
    print(f"  {word:15s}: {count:6,} occurrences")

print("\n" + "="*80)

# Train Word2Vec model
print("\nTRAINING WORD2VEC MODEL:")
print("-" * 80)

print(f"Training Word2Vec...")
print(f"  Embedding dimension: {CONFIG['embedding_dim']}")
print(f"  Window size: 5")
print(f"  Min count: {CONFIG['min_word_frequency']}")

w2v_model = Word2Vec(
    sentences=processed_texts,
    vector_size=CONFIG['embedding_dim'],
    window=5,
    min_count=CONFIG['min_word_frequency'],
    workers=4,
    sg=0,  # CBOW
    epochs=10
)

print(f"✓ Word2Vec model trained")
print(f"  Vocabulary size: {len(w2v_model.wv):,}")

# Create embedding matrix
print(f"\nCreating embedding matrix...")
embedding_matrix = np.zeros((CONFIG['vocab_size'], CONFIG['embedding_dim']))

embeddings_found = 0
for word, idx in word2idx.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
        embeddings_found += 1
    else:
        # Initialize with small random values
        embedding_matrix[idx] = np.random.normal(0, 0.1, CONFIG['embedding_dim'])

print(f"✓ Embedding matrix created: shape {embedding_matrix.shape}")
print(f"  Embeddings found: {embeddings_found}/{CONFIG['vocab_size']} ({embeddings_found/CONFIG['vocab_size']*100:.2f}%)")

print("\n" + "="*80)

# Convert texts to sequences
print("\nCONVERTING TEXTS TO SEQUENCES:")
print("-" * 80)

def texts_to_sequences(texts, word2idx, max_length):
    """Convert list of token lists to padded sequences."""
    sequences = []
    for tokens in texts:
        # Convert tokens to indices
        seq = [word2idx.get(token, 0) for token in tokens]
        
        # Pad or truncate
        if len(seq) < max_length:
            seq = seq + [0] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        
        sequences.append(seq)
    
    return np.array(sequences)

X_sequences = texts_to_sequences(
    processed_texts,
    word2idx,
    CONFIG['max_sequence_length']
)

y_labels = df_combined['label'].values

print(f"✓ Sequences created:")
print(f"  X shape: {X_sequences.shape}")
print(f"  y shape: {y_labels.shape}")
print(f"  Sequence length: {CONFIG['max_sequence_length']}")

print("\n" + "="*80)

# Save all preprocessed data
print("\nSAVING PREPROCESSED DATA:")
print("-" * 80)

# Save numpy arrays
np.save(os.path.join(PREPROCESSED_DIR, 'X_sequences.npy'), X_sequences)
print(f"✓ Saved: X_sequences.npy")

np.save(os.path.join(PREPROCESSED_DIR, 'y_labels.npy'), y_labels)
print(f"✓ Saved: y_labels.npy")

np.save(os.path.join(PREPROCESSED_DIR, 'embedding_matrix.npy'), embedding_matrix)
print(f"✓ Saved: embedding_matrix.npy")

# Save vocabulary mappings
with open(os.path.join(PREPROCESSED_DIR, 'word2idx.pkl'), 'wb') as f:
    pickle.dump(word2idx, f)
print(f"✓ Saved: word2idx.pkl")

with open(os.path.join(PREPROCESSED_DIR, 'idx2word.pkl'), 'wb') as f:
    pickle.dump(idx2word, f)
print(f"✓ Saved: idx2word.pkl")

# Save configuration
with open(os.path.join(PREPROCESSED_DIR, 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=4)
print(f"✓ Saved: config.json")

# Save Word2Vec model
w2v_model.save(os.path.join(PREPROCESSED_DIR, 'word2vec_model.bin'))
print(f"✓ Saved: word2vec_model.bin")

print("\n" + "="*80)
print("✨ PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\nPreprocessed files saved to: {PREPROCESSED_DIR}")
print(f"\nNext step: Run the training script (2_train_model.py)")
