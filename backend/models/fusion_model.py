"""
Multimodal Emotion Fusion Model
Combines facial and text emotion predictions with incongruence detection
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class EmotionPrediction:
    """Structure for emotion prediction results"""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

class MultimodalFusion:
    """
    Fuses facial and text emotion predictions
    Implements weighted fusion and incongruence detection
    """
    
    # Unified emotion set (6 core emotions + neutral)
    EMOTIONS = ['Happy', 'Sad', 'Angry', 'Fearful', 'Surprised', 'Neutral']
    
    # Emotion mapping from different models
    FACIAL_TO_UNIFIED = {
        'Happy': 'Happy',
        'Sad': 'Sad',
        'Angry': 'Angry',
        'Fearful': 'Fearful',
        'Surprised': 'Surprised',
        'Neutral': 'Neutral',
        'Disgusted': 'Angry',  # Map disgust to anger
    }
    
    TEXT_TO_UNIFIED = {
        'joy': 'Happy',
        'sadness': 'Sad',
        'anger': 'Angry',
        'fear': 'Fearful',
        'surprise': 'Surprised',
        'love': 'Happy',  # Map love to happy
    }
    
    # Emotion opposition matrix for incongruence detection
    OPPOSING_EMOTIONS = {
        'Happy': ['Sad', 'Angry', 'Fearful'],
        'Sad': ['Happy', 'Surprised'],
        'Angry': ['Happy'],
        'Fearful': ['Happy'],
        'Surprised': [],
        'Neutral': []
    }
    
    def __init__(
        self,
        facial_model,
        text_model,
        text_weight: float = 0.6,
        facial_weight: float = 0.4,
        incongruence_threshold: float = 0.7
    ):
        """
        Initialize fusion model
        
        Args:
            facial_model: Trained facial emotion model
            text_model: Trained text emotion model
            text_weight: Weight for text predictions (default: 0.6)
            facial_weight: Weight for facial predictions (default: 0.4)
            incongruence_threshold: Confidence threshold for incongruence (default: 0.7)
        """
        self.facial_model = facial_model
        self.text_model = text_model
        self.text_weight = text_weight
        self.facial_weight = facial_weight
        self.incongruence_threshold = incongruence_threshold
        
        # Validate weights
        assert abs(text_weight + facial_weight - 1.0) < 0.001, "Weights must sum to 1.0"
    
    def fuse(
        self,
        text_emotion: Optional[str] = None,
        text_confidence: Optional[float] = None,
        facial_emotion: Optional[str] = None,
        facial_confidence: Optional[float] = None
    ) -> Dict:
        """
        Fuse emotion predictions from multiple modalities
        
        Args:
            text_emotion: Predicted emotion from text
            text_confidence: Confidence score for text prediction
            facial_emotion: Predicted emotion from facial analysis
            facial_confidence: Confidence score for facial prediction
        
        Returns:
            Dictionary containing fused emotion and analysis
        """
        
        # Case 1: Both modalities available
        if text_emotion and facial_emotion:
            return self._fuse_both_modalities(
                text_emotion, text_confidence,
                facial_emotion, facial_confidence
            )
        
        # Case 2: Only text available
        elif text_emotion:
            return self._text_only_result(text_emotion, text_confidence)
        
        # Case 3: Only facial available
        elif facial_emotion:
            return self._facial_only_result(facial_emotion, facial_confidence)
        
        # Case 4: No data (should not happen)
        else:
            return self._neutral_result()
    
    def _fuse_both_modalities(
        self,
        text_emotion: str,
        text_confidence: float,
        facial_emotion: str,
        facial_confidence: float
    ) -> Dict:
        """Fuse predictions when both modalities are available"""
        
        # Normalize emotions to unified set
        text_unified = self.TEXT_TO_UNIFIED.get(text_emotion, text_emotion)
        facial_unified = self.FACIAL_TO_UNIFIED.get(facial_emotion, facial_emotion)
        
        # Check for incongruence
        incongruence_detected = self._detect_incongruence(
            text_unified, text_confidence,
            facial_unified, facial_confidence
        )
        
        # Weighted fusion
        if text_unified == facial_unified:
            # Agreement: use combined confidence
            final_emotion = text_unified
            final_confidence = (
                text_confidence * self.text_weight +
                facial_confidence * self.facial_weight
            )
            fusion_method = "agreement"
            
        else:
            # Disagreement: use confidence-weighted decision
            text_score = text_confidence * self.text_weight
            facial_score = facial_confidence * self.facial_weight
            
            if text_score > facial_score:
                final_emotion = text_unified
                final_confidence = text_confidence
                fusion_method = "text_dominant"
            else:
                final_emotion = facial_unified
                final_confidence = facial_confidence
                fusion_method = "facial_dominant"
        
        result = {
            'final_emotion': final_emotion,
            'confidence': float(final_confidence),
            'text_emotion': text_unified,
            'text_confidence': float(text_confidence),
            'facial_emotion': facial_unified,
            'facial_confidence': float(facial_confidence),
            'fusion_method': fusion_method,
            'incongruence': incongruence_detected,
            'modalities_used': ['text', 'facial']
        }
        
        # Add incongruence message if detected
        if incongruence_detected:
            result['incongruence_message'] = self._generate_incongruence_message(
                text_unified, facial_unified
            )
        
        return result
    
    def _detect_incongruence(
        self,
        text_emotion: str,
        text_confidence: float,
        facial_emotion: str,
        facial_confidence: float
    ) -> bool:
        """
        Detect emotional incongruence between modalities
        
        Incongruence is detected when:
        1. Emotions are opposing (e.g., Happy text but Sad face)
        2. Both predictions have high confidence (> threshold)
        
        This is clinically significant as it may indicate:
        - Masking of true emotions
        - Social desirability bias
        - Alexithymia (difficulty identifying emotions)
        - Emotional suppression
        """
        
        # Check if emotions are different
        if text_emotion == facial_emotion:
            return False
        
        # Check if both have high confidence
        if (text_confidence < self.incongruence_threshold or 
            facial_confidence < self.incongruence_threshold):
            return False
        
        # Check if emotions are opposing
        if facial_emotion in self.OPPOSING_EMOTIONS.get(text_emotion, []):
            return True
        
        return False
    
    def _generate_incongruence_message(
        self,
        text_emotion: str,
        facial_emotion: str
    ) -> str:
        """Generate helpful message about detected incongruence"""
        
        messages = {
            ('Happy', 'Sad'): "I notice you're expressing positive feelings in your words, but your facial expression seems sad. It's okay to share how you truly feel.",
            ('Happy', 'Angry'): "Your words seem upbeat, but I sense some tension in your expression. Would you like to talk about what might be bothering you?",
            ('Happy', 'Fearful'): "While your message sounds positive, I notice some concern in your expression. Is something worrying you?",
            ('Sad', 'Happy'): "I hear sadness in your words, but your expression seems more positive. Sometimes we put on a brave face - would you like to explore those feelings?",
            ('Sad', 'Neutral'): "You mentioned feeling sad, but your expression seems more neutral. Sometimes it's hard to show our emotions - I'm here to listen.",
            ('Angry', 'Happy'): "I'm picking up on some frustration in your words, though your expression seems different. Would you like to talk about what's bothering you?",
        }
        
        key = (text_emotion, facial_emotion)
        return messages.get(key, 
            f"I notice your words express {text_emotion}, while your facial expression shows {facial_emotion}. "
            "Sometimes our feelings are complex. Would you like to explore this further?"
        )
    
    def _text_only_result(self, emotion: str, confidence: float) -> Dict:
        """Return result when only text is available"""
        
        unified = self.TEXT_TO_UNIFIED.get(emotion, emotion)
        
        return {
            'final_emotion': unified,
            'confidence': float(confidence),
            'text_emotion': unified,
            'text_confidence': float(confidence),
            'facial_emotion': None,
            'facial_confidence': 0.0,
            'fusion_method': 'text_only',
            'incongruence': False,
            'modalities_used': ['text']
        }
    
    def _facial_only_result(self, emotion: str, confidence: float) -> Dict:
        """Return result when only facial is available"""
        
        unified = self.FACIAL_TO_UNIFIED.get(emotion, emotion)
        
        return {
            'final_emotion': unified,
            'confidence': float(confidence),
            'text_emotion': None,
            'text_confidence': 0.0,
            'facial_emotion': unified,
            'facial_confidence': float(confidence),
            'fusion_method': 'facial_only',
            'incongruence': False,
            'modalities_used': ['facial']
        }
    
    def _neutral_result(self) -> Dict:
        """Return neutral result when no data is available"""
        
        return {
            'final_emotion': 'Neutral',
            'confidence': 0.0,
            'text_emotion': None,
            'text_confidence': 0.0,
            'facial_emotion': None,
            'facial_confidence': 0.0,
            'fusion_method': 'no_data',
            'incongruence': False,
            'modalities_used': []
        }
    
    def get_emotion_vector(self, emotion_result: Dict) -> np.ndarray:
        """
        Convert emotion result to vector representation
        Useful for further processing or visualization
        """
        
        vector = np.zeros(len(self.EMOTIONS))
        emotion = emotion_result['final_emotion']
        confidence = emotion_result['confidence']
        
        if emotion in self.EMOTIONS:
            idx = self.EMOTIONS.index(emotion)
            vector[idx] = confidence
        
        return vector
    
    def analyze_session_emotions(self, emotion_history: list) -> Dict:
        """
        Analyze emotion patterns across a session
        
        Args:
            emotion_history: List of emotion results from the session
        
        Returns:
            Dictionary with session-level insights
        """
        
        if not emotion_history:
            return {'error': 'No emotion history available'}
        
        # Extract emotions
        emotions = [e['final_emotion'] for e in emotion_history]
        
        # Count occurrences
        from collections import Counter
        emotion_counts = Counter(emotions)
        
        # Calculate emotion percentages
        total = len(emotions)
        emotion_percentages = {
            emotion: (count / total) * 100
            for emotion, count in emotion_counts.items()
        }
        
        # Find dominant emotion
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Count incongruences
        incongruence_count = sum(1 for e in emotion_history if e.get('incongruence', False))
        incongruence_rate = (incongruence_count / total) * 100
        
        # Emotion trajectory (simplified)
        trajectory = [e['final_emotion'] for e in emotion_history[-10:]]  # Last 10
        
        # Average confidence
        avg_confidence = np.mean([e['confidence'] for e in emotion_history])
        
        return {
            'total_messages': total,
            'dominant_emotion': dominant_emotion,
            'emotion_distribution': emotion_percentages,
            'incongruence_count': incongruence_count,
            'incongruence_rate': float(incongruence_rate),
            'recent_trajectory': trajectory,
            'average_confidence': float(avg_confidence),
            'analysis': self._generate_session_insights(
                dominant_emotion, incongruence_rate, emotion_percentages
            )
        }
    
    def _generate_session_insights(
        self,
        dominant_emotion: str,
        incongruence_rate: float,
        emotion_distribution: Dict[str, float]
    ) -> str:
        """Generate human-readable session insights"""
        
        insights = []
        
        # Dominant emotion insight
        if dominant_emotion == 'Sad':
            insights.append("The conversation shows predominant sadness. This may indicate the user is experiencing low mood or depression.")
        elif dominant_emotion == 'Angry':
            insights.append("Anger is the most frequent emotion. The user may be experiencing frustration or irritation.")
        elif dominant_emotion == 'Fearful':
            insights.append("Fear is prominent. The user may be experiencing anxiety or worry.")
        elif dominant_emotion == 'Happy':
            insights.append("The conversation shows mostly positive emotions.")
        
        # Incongruence insight
        if incongruence_rate > 30:
            insights.append(
                f"High emotional incongruence detected ({incongruence_rate:.1f}%). "
                "The user may be masking their true feelings or experiencing difficulty expressing emotions."
            )
        elif incongruence_rate > 15:
            insights.append(
                f"Moderate emotional incongruence detected ({incongruence_rate:.1f}%). "
                "Some instances where expressed and felt emotions may differ."
            )
        
        # Emotion variety
        if len(emotion_distribution) == 1:
            insights.append("Emotional range is limited. Consider exploring different aspects of the user's experience.")
        elif len(emotion_distribution) >= 4:
            insights.append("Wide emotional range observed. This suggests the user is experiencing complex or mixed feelings.")
        
        return " ".join(insights) if insights else "Session analysis in progress."


# Example usage
if __name__ == "__main__":
    # Mock models for testing
    class MockModel:
        def predict(self, x):
            return "Happy", 0.85
    
    facial_model = MockModel()
    text_model = MockModel()
    
    fusion = MultimodalFusion(facial_model, text_model)
    
    # Test fusion
    result = fusion.fuse(
        text_emotion='joy',
        text_confidence=0.92,
        facial_emotion='Sad',
        facial_confidence=0.88
    )
    
    print("Fusion Result:")
    print(result)