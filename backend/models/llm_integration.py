"""
LLM Integration for Therapeutic Response Generation
Uses OpenAI GPT to generate empathetic, context-aware responses
"""

# import openai
from typing import List, Dict, Optional
import json
from datetime import datetime
from openai import AsyncOpenAI


class TherapeuticLLM:
    """
    Generates therapeutic responses using OpenAI's GPT models
    Integrates emotion detection for emotionally intelligent conversations
    """
    
    # Emotion-specific therapeutic prompting strategies
    EMOTION_STRATEGIES = {
        'Happy': {
            'approach': 'Supportive and encouraging',
            'goals': ['Reinforce positive feelings', 'Explore sources of happiness', 'Build on strengths']
        },
        'Sad': {
            'approach': 'Empathetic and validating',
            'goals': ['Validate feelings', 'Explore underlying causes', 'Offer hope and support']
        },
        'Angry': {
            'approach': 'Calm and understanding',
            'goals': ['Acknowledge anger as valid', 'Explore triggers', 'Discuss healthy expression']
        },
        'Fearful': {
            'approach': 'Reassuring and grounding',
            'goals': ['Provide reassurance', 'Explore fears safely', 'Discuss coping strategies']
        },
        'Surprised': {
            'approach': 'Curious and exploratory',
            'goals': ['Understand the surprise', 'Process unexpected feelings', 'Support adjustment']
        },
        'Neutral': {
            'approach': 'Gentle and exploratory',
            'goals': ['Encourage expression', 'Build rapport', 'Explore beneath surface']
        }
    }
    
    # Base system prompt for therapeutic mode
    BASE_SYSTEM_PROMPT = """You are an empathetic AI mental health counselor. Your role is to:

1. Provide emotional support and validation
2. Use active listening and reflective responses
3. Ask thoughtful, open-ended questions
4. Maintain appropriate boundaries (you're not a crisis hotline)
5. Encourage professional help when needed
6. Never judge or give medical advice
7. Create a safe, non-judgmental space

Your responses should be:
- Warm and compassionate
- Conversational (not overly formal)
- Concise (2-4 sentences typically)
- Focused on the person's emotional experience
- Trauma-informed and culturally sensitive

Remember: You're here to listen, validate, and gently guide - not to solve or diagnose."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 200
    ):
        """
        Initialize LLM service
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4-turbo-preview recommended)
            temperature: Response randomness (0.7 for balanced creativity)
            max_tokens: Maximum response length
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate_response(
        self,
        user_message: str,
        detected_emotion: str,
        emotional_context: Dict,
        session_history: List[Dict] = None,
        user_context: Dict = None
    ) -> str:
        """
        Generate therapeutic response based on emotion and context
        
        Args:
            user_message: The user's text input
            detected_emotion: Primary emotion detected
            emotional_context: Full emotion analysis including incongruence
            session_history: Previous conversation history
            user_context: Additional context about the user
        
        Returns:
            Therapeutic response string
        """
        
        # Build conversation context
        messages = self._build_message_context(
            user_message=user_message,
            detected_emotion=detected_emotion,
            emotional_context=emotional_context,
            session_history=session_history or [],
            user_context=user_context or {}
        )
        
        # Generate response
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                presence_penalty=0.6,  # Encourage diverse responses
                frequency_penalty=0.3   # Reduce repetition
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._get_fallback_response(detected_emotion)
    
    def _build_message_context(
        self,
        user_message: str,
        detected_emotion: str,
        emotional_context: Dict,
        session_history: List[Dict],
        user_context: Dict
    ) -> List[Dict[str, str]]:
        """Build the message array for the LLM"""
        
        # System message with emotion-specific guidance
        system_prompt = self._create_system_prompt(detected_emotion, emotional_context)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add relevant history (last 5 exchanges)
        recent_history = session_history[-10:] if len(session_history) > 10 else session_history
        for exchange in recent_history:
            messages.append({
                "role": "user",
                "content": exchange.get('user_message', '')
            })
            messages.append({
                "role": "assistant",
                "content": exchange.get('bot_response', '')
            })
        
        # Add current message with emotion annotation
        emotion_note = self._create_emotion_annotation(detected_emotion, emotional_context)
        current_message = f"{user_message}\n\n{emotion_note}"
        
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    def _create_system_prompt(self, emotion: str, emotional_context: Dict) -> str:
        """Create emotion-specific system prompt"""
        
        strategy = self.EMOTION_STRATEGIES.get(emotion, self.EMOTION_STRATEGIES['Neutral'])
        
        emotion_guidance = f"""
Current Emotional State: {emotion}
Therapeutic Approach: {strategy['approach']}
Session Goals: {', '.join(strategy['goals'])}

"""
        
        # Add incongruence guidance if detected
        if emotional_context.get('incongruence'):
            incongruence_guidance = f"""
⚠️ EMOTIONAL INCONGRUENCE DETECTED:
- Verbal emotion: {emotional_context.get('text_emotion')}
- Facial emotion: {emotional_context.get('facial_emotion')}

This suggests the person may be masking their true feelings. Approach with extra sensitivity:
- Gently acknowledge the complexity of their emotions
- Create space for them to share what they're truly feeling
- Don't confront directly - use reflective statements
- Example: "It sounds like you're dealing with some complex feelings right now..."

"""
            emotion_guidance += incongruence_guidance
        
        return self.BASE_SYSTEM_PROMPT + "\n\n" + emotion_guidance
    
    def _create_emotion_annotation(self, emotion: str, context: Dict) -> str:
        """Create internal emotion annotation for the LLM"""
        
        annotation = f"[Internal Context: User emotion detected as {emotion}"
        
        confidence = context.get('confidence', 0)
        if confidence > 0.8:
            annotation += " (high confidence)"
        elif confidence < 0.5:
            annotation += " (uncertain)"
        
        if context.get('incongruence'):
            annotation += f". Emotional incongruence: text={context.get('text_emotion')}, facial={context.get('facial_emotion')}"
        
        annotation += "]"
        
        return annotation
    
    def _get_fallback_response(self, emotion: str) -> str:
        """Fallback responses if LLM fails"""
        
        fallback_responses = {
            'Happy': "I'm glad to hear you're feeling positive. Tell me more about what's bringing you joy today.",
            'Sad': "I hear that you're going through a difficult time. I'm here to listen. Would you like to talk about what's troubling you?",
            'Angry': "It sounds like something has really upset you. Your feelings are valid. Would you like to share what's bothering you?",
            'Fearful': "I sense you might be feeling worried or anxious. That's completely understandable. What's on your mind?",
            'Surprised': "It seems like something unexpected has happened. How are you processing that?",
            'Neutral': "I'm here to listen and support you. What's on your mind today?"
        }
        
        return fallback_responses.get(emotion, "I'm here to listen. How are you feeling?")
    
    def generate_opening_message(self, user_name: Optional[str] = None) -> str:
        """Generate personalized opening message"""
        
        if user_name:
            return f"Hello {user_name}, I'm here to listen and support you. How are you feeling today?"
        else:
            return "Hello, I'm here to provide a safe space for you to share your thoughts and feelings. How can I support you today?"
    
    def generate_closing_message(self, session_summary: Dict) -> str:
        """Generate session closing message with summary"""
        
        dominant_emotion = session_summary.get('dominant_emotion', 'Neutral')
        
        closings = {
            'Happy': "I'm glad we could talk while you're feeling positive. Remember to hold onto these feelings. Take care, and I'm here whenever you need support.",
            'Sad': "Thank you for sharing your feelings with me today. Remember, it's okay to not be okay. I'm here whenever you need someone to talk to. Be gentle with yourself.",
            'Angry': "I appreciate you sharing your frustrations with me. Remember to take care of yourself, and I'm here if you need to talk more. Take some deep breaths.",
            'Fearful': "Thank you for trusting me with your worries. Remember, you're stronger than you think. I'm here for you. Take things one step at a time.",
            'Surprised': "Thank you for sharing your experience. Take time to process what's happened. I'm here if you need to talk more.",
            'Neutral': "Thank you for our conversation today. Remember, I'm here whenever you need support. Take care of yourself."
        }
        
        return closings.get(dominant_emotion, closings['Neutral'])
    
    async def generate_crisis_response(self) -> str:
        """Generate response for crisis situations"""
        
        return """I'm concerned about what you're sharing. While I'm here to support you, it's important that you speak with a qualified professional who can provide immediate help.

Please reach out to:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

If you're in immediate danger, please call emergency services (911 in US).

I care about your wellbeing and want to ensure you get the help you need."""
    
    def detect_crisis_keywords(self, message: str) -> bool:
        """Detect potential crisis situations"""
        
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'hurt myself', 'self harm', 'overdose', 'no reason to live',
            'better off dead', 'suicidal'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in crisis_keywords)
    
    async def generate_summary(self, session_history: List[Dict]) -> str:
        """Generate session summary for user or clinician"""
        
        if not session_history:
            return "No session data available."
        
        # Extract key information
        messages = [ex.get('user_message', '') for ex in session_history]
        emotions = [ex.get('emotion_data', {}).get('final_emotion', 'Unknown') 
                   for ex in session_history]
        
        summary_prompt = f"""Generate a brief, compassionate summary of this therapy session:

Number of exchanges: {len(session_history)}
Emotions detected: {', '.join(set(emotions))}

Recent messages:
{chr(10).join(messages[-5:])}

Provide a 2-3 sentence summary focusing on:
1. Main themes discussed
2. Emotional state progression
3. Any concerns or positive developments"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical psychologist summarizing a therapy session."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Session with {len(session_history)} exchanges, primarily discussing {', '.join(set(emotions))} emotions."


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        llm = TherapeuticLLM(api_key="your-api-key-here")
        
        # Test response generation
        response = await llm.generate_response(
            user_message="I've been feeling really down lately",
            detected_emotion="Sad",
            emotional_context={
                'final_emotion': 'Sad',
                'confidence': 0.85,
                'incongruence': False
            },
            session_history=[]
        )
        
        print("Generated Response:")
        print(response)
    
    asyncio.run(test())