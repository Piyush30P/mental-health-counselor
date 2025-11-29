"""
Facial Emotion Recognition Model
Wrapper for your trained CNN-LSTM facial emotion model
"""

import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FacialEmotionModel:
    """
    Facial emotion recognition using CNN-LSTM architecture
    Trained on CK+ dataset with 7 emotion classes
    """
    
    # Emotion labels (from your trained model)
    EMOTION_LABELS = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Sad",
        5: "Surprised",
        6: "Neutral"
    }
    
    def __init__(self, model_path: str):
        """
        Initialize facial emotion model
        
        Args:
            model_path: Path to trained .h5 model file
        """
        logger.info(f"Loading facial emotion model from: {model_path}")
        
        try:
            # Load your trained model
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Input shape: {self.model.input_shape}")
            logger.info(f"  Output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load facial model: {e}")
            raise
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade")
            raise ValueError("Could not load face detection cascade")
        
        logger.info("✓ Face detection cascade loaded")
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame using Haar Cascade
        
        Args:
            frame: BGR image from camera
            
        Returns:
            (x, y, w, h) of detected face or None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def preprocess_face(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and preprocess face ROI for model input
        
        Args:
            frame: BGR image
            face_coords: (x, y, w, h) face coordinates
            
        Returns:
            Preprocessed face array ready for model
        """
        x, y, w, h = face_coords
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to 48x48 (your model's input size)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        
        # Normalize to [0, 1]
        roi_normalized = roi_resized / 255.0
        
        # Reshape for model: (1, 48, 48, 1)
        # Batch size=1, Height=48, Width=48, Channels=1 (grayscale)
        roi_input = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)
        
        return roi_input
    
    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Predict emotion from a single frame
        
        Args:
            frame: BGR image from webcam (any size)
            
        Returns:
            (emotion_name, confidence) tuple
            Returns ("Neutral", 0.0) if no face detected
        """
        try:
            # Detect face
            face_coords = self.detect_face(frame)
            
            if face_coords is None:
                logger.debug("No face detected in frame")
                return "Neutral", 0.0
            
            # Preprocess face
            face_input = self.preprocess_face(frame, face_coords)
            
            # Predict emotion
            predictions = self.model.predict(face_input, verbose=0)
            
            # Get emotion with highest probability
            emotion_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][emotion_idx])
            
            emotion_name = self.EMOTION_LABELS[emotion_idx]
            
            logger.debug(f"Predicted: {emotion_name} ({confidence:.2f})")
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"Error in facial emotion prediction: {e}")
            return "Neutral", 0.0
    
    def predict_with_visualization(self, frame: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion and return annotated frame
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            (emotion_name, confidence, annotated_frame)
        """
        annotated_frame = frame.copy()
        
        # Detect face
        face_coords = self.detect_face(frame)
        
        if face_coords is None:
            return "Neutral", 0.0, annotated_frame
        
        x, y, w, h = face_coords
        
        # Draw rectangle around face
        cv2.rectangle(
            annotated_frame,
            (x, y-50),
            (x+w, y+h+10),
            (255, 0, 0),
            2
        )
        
        # Get prediction
        emotion, confidence = self.predict(frame)
        
        # Add text
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(
            annotated_frame,
            text,
            (x+20, y-60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return emotion, confidence, annotated_frame
    
    def predict_batch(self, frames: list) -> list:
        """
        Predict emotions for multiple frames
        
        Args:
            frames: List of BGR images
            
        Returns:
            List of (emotion, confidence) tuples
        """
        results = []
        for frame in frames:
            emotion, confidence = self.predict(frame)
            results.append((emotion, confidence))
        return results


# Test function
def test_facial_model():
    """Test the facial emotion model with webcam"""
    import time
    
    print("Testing Facial Emotion Model...")
    print("=" * 60)
    
    # Initialize model
    model = FacialEmotionModel("weights/facial_model.h5")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened. Press 'q' to quit.")
    print()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Predict with visualization
        emotion, confidence, annotated = model.predict_with_visualization(frame)
        
        # Display result
        cv2.imshow('Facial Emotion Recognition - Press Q to Quit', annotated)
        
        frame_count += 1
        
        # Print stats every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frames: {frame_count} | FPS: {fps:.2f} | Emotion: {emotion} ({confidence:.2f})")
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print(f"Test completed. Processed {frame_count} frames.")


if __name__ == "__main__":
    # Run test if executed directly
    test_facial_model()