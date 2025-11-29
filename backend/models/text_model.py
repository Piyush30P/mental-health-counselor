"""
Text Emotion Recognition Model
Wrapper for your trained CNN-BiLSTM text emotion model
"""

import tensorflow as tf
import numpy as np
import pickle
import json
from typing import Tuple, List
import re
import string
import logging

# NLP imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logger = logging.getLogger(__name__)


class TextEmotionModel:
    """
    Text emotion recognition using CNN-BiLSTM architecture
    Trained on emotion datasets with 6 emotion classes
    """
    
    # Emotion labels (from your trained model)
    EMOTION_LABELS = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }
    
    # Contraction mapping for text cleaning
    CONTRACTION_MAP = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot",
        "can't've": "cannot have", "could've": "could have",
        "couldn't": "could not", "didn't": "did not",
        "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would",
        "he'll": "he will", "he's": "he is",
        "i'd": "I would", "i'll": "I will",
        "i'm": "I am", "i've": "I have",
        "isn't": "is not", "it'd": "it would",
        "it'll": "it will", "it's": "it is",
        "let's": "let us", "mustn't": "must not",
        "shan't": "shall not", "she'd": "she would",
        "she'll": "she will", "she's": "she is",
        "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would",
        "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not",
        "we'd": "we would", "we'll": "we will",
        "we're": "we are", "we've": "we have",
        "weren't": "were not", "what'll": "what will",
        "what're": "what are", "what's": "what is",
        "what've": "what have", "where's": "where is",
        "who'll": "who will", "who's": "who is",
        "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will",
        "you're": "you are", "you've": "you have"
    }
    
    def __init__(self, model_path: str, vocab_path: str, config_path: str):
        """
        Initialize text emotion model
        
        Args:
            model_path: Path to trained .h5 model file
            vocab_path: Path to word2idx.pkl vocabulary file
            config_path: Path to config.json configuration file
        """
        logger.info(f"Loading text emotion model from: {model_path}")
        
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load text model: {e}")
            raise
        
        # Load vocabulary
        try:
            with open(vocab_path, 'rb') as f:
                self.word2idx = pickle.load(f)
            logger.info(f"✓ Vocabulary loaded: {len(self.word2idx):,} words")
            
        except Exception as e:
            logger.error(f"✗ Failed to load vocabulary: {e}")
            raise
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"✓ Configuration loaded")
            
        except Exception as e:
            logger.error(f"✗ Failed to load config: {e}")
            raise
        
        self.max_length = self.config.get('max_sequence_length', 100)
        logger.info(f"  Max sequence length: {self.max_length}")
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Download NLTK data if needed
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data packages"""
        packages = [
            'punkt',
            'stopwords',
            'wordnet',
            'omw-1.4',
            'averaged_perceptron_tagger'
        ]
        
        for package in packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK package: {package}")
                    nltk.download(package, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download {package}: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and standardize text
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Expand contractions
        for contraction, expansion in self.CONTRACTION_MAP.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convert TreeBank POS tag to WordNet POS tag
        
        Args:
            treebank_tag: POS tag from NLTK
            
        Returns:
            WordNet POS tag
        """
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
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed sequence ready for model
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Lowercase
        tokens = [token.lower() for token in tokens]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # POS tagging and lemmatization
        pos_tags = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]
        
        # Convert to indices
        indices = [self.word2idx.get(token, 0) for token in lemmatized]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        # Convert to numpy array with batch dimension
        return np.array([indices])
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict emotion from text
        
        Args:
            text: Input text string
            
        Returns:
            (emotion_name, confidence) tuple
        """
        try:
            # Preprocess text
            processed = self.preprocess_text(text)
            
            # Predict
            predictions = self.model.predict(processed, verbose=0)
            
            # Get emotion with highest probability
            emotion_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][emotion_idx])
            
            emotion_name = self.EMOTION_LABELS[emotion_idx]
            
            logger.debug(f"Text predicted: {emotion_name} ({confidence:.2f})")
            logger.debug(f"  Input: {text[:50]}...")
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"Error in text emotion prediction: {e}")
            return "neutral", 0.0
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict emotions for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (emotion, confidence) tuples
        """
        results = []
        for text in texts:
            emotion, confidence = self.predict(text)
            results.append((emotion, confidence))
        return results
    
    def get_emotion_probabilities(self, text: str) -> dict:
        """
        Get probability distribution over all emotions
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emotion names to probabilities
        """
        processed = self.preprocess_text(text)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        probabilities = {
            self.EMOTION_LABELS[i]: float(predictions[i])
            for i in range(len(self.EMOTION_LABELS))
        }
        
        return probabilities


# Test function
def test_text_model():
    """Test the text emotion model"""
    print("Testing Text Emotion Model...")
    print("=" * 60)
    
    # Initialize model
    model = TextEmotionModel(
        "weights/text_model.h5",
        "weights/word2idx.pkl",
        "weights/config.json"
    )
    
    # Test sentences
    test_sentences = [
        "I am so happy and excited about this!",
        "This makes me really angry and frustrated.",
        "I feel so sad and lonely today.",
        "I love spending time with my family.",
        "That movie was really scary and terrifying.",
        "Wow! I can't believe this happened!",
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    for i, text in enumerate(test_sentences, 1):
        emotion, confidence = model.predict(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Emotion: {emotion.upper()} ({confidence*100:.1f}%)")
        
        # Get full probability distribution
        probs = model.get_emotion_probabilities(text)
        print("   All probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 50)
            print(f"     {emo:10s}: {bar} {prob*100:.1f}%")
    
    print()
    print("=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    # Run test if executed directly
    test_text_model()