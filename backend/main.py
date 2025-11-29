"""
Mental Health Counselor - FastAPI Backend
Complete multimodal emotion recognition with authentication and database
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
from typing import Dict, Optional
import asyncio
from datetime import datetime
import base64
import logging
import sys

# Import configuration
from config import settings

# Import models
from models.facial_model import FacialEmotionModel
from models.text_model import TextEmotionModel
from models.fusion_model import MultimodalFusion
from models.llm_integration import TherapeuticLLM

# Import database
from database.connection import MongoDB, get_sessions_collection

# Import routers
from routers import auth, user, admin

# Import auth dependencies
from auth.dependencies import get_current_active_user

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global model instances
facial_model: Optional[FacialEmotionModel] = None
text_model: Optional[TextEmotionModel] = None
fusion_model: Optional[MultimodalFusion] = None
llm_service: Optional[TherapeuticLLM] = None

# Session storage (in-memory for active sessions)
active_sessions: Dict[str, dict] = {}
MAX_SESSIONS = settings.MAX_SESSIONS


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global facial_model, text_model, fusion_model, llm_service
    
    logger.info("="*80)
    logger.info("MENTAL HEALTH COUNSELOR - STARTING UP")
    logger.info("="*80)
    
    try:
        # Connect to MongoDB
        logger.info("🗄️  Connecting to MongoDB...")
        await MongoDB.connect_db()
        logger.info("✓ MongoDB connected successfully")
        
        # Load facial emotion model
        logger.info("📹 Loading facial emotion model...")
        facial_model = FacialEmotionModel(settings.FACIAL_MODEL_PATH)
        logger.info("✓ Facial model loaded successfully")
        
        # Load text emotion model
        logger.info("💬 Loading text emotion model...")
        text_model = TextEmotionModel(
            settings.TEXT_MODEL_PATH,
            settings.VOCAB_PATH,
            settings.CONFIG_PATH
        )
        logger.info("✓ Text model loaded successfully")
        
        # Initialize fusion model
        logger.info("🔀 Initializing multimodal fusion...")
        fusion_model = MultimodalFusion(
            facial_model=facial_model,
            text_model=text_model,
            text_weight=settings.TEXT_WEIGHT,
            facial_weight=settings.FACIAL_WEIGHT,
            incongruence_threshold=settings.INCONGRUENCE_THRESHOLD
        )
        logger.info(f"✓ Fusion model initialized (Text: {settings.TEXT_WEIGHT}, Facial: {settings.FACIAL_WEIGHT})")
        
        # Initialize LLM service
        if settings.OPENAI_API_KEY:
            logger.info("🤖 Initializing therapeutic LLM...")
            llm_service = TherapeuticLLM(
                api_key=settings.OPENAI_API_KEY,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            logger.info("✓ LLM service initialized")
        else:
            logger.warning("⚠️  OpenAI API key not found - LLM responses will be limited")
            llm_service = None
        
        logger.info("="*80)
        logger.info("✨ ALL SYSTEMS READY - Server starting...")
        logger.info("="*80)
        
        yield
        
        # Shutdown
        await MongoDB.close_db()
        active_sessions.clear()
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ ERROR during startup: {e}")
        raise


# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Counselor API",
    description="Multimodal emotion recognition with therapeutic AI responses",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(admin.router)


# ============= ROOT & HEALTH ENDPOINTS =============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Mental Health Counselor API",
        "version": "2.0.0",
        "models_loaded": {
            "facial": facial_model is not None,
            "text": text_model is not None,
            "fusion": fusion_model is not None,
            "llm": llm_service is not None
        },
        "database": "connected" if MongoDB.database else "disconnected",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "facial_model": "loaded" if facial_model else "not loaded",
            "text_model": "loaded" if text_model else "not loaded",
            "fusion_model": "loaded" if fusion_model else "not loaded",
            "llm_service": "loaded" if llm_service else "not loaded"
        },
        "database": {
            "status": "connected" if MongoDB.database else "disconnected",
            "name": MongoDB.database.name if MongoDB.database else None
        },
        "sessions": {
            "active": len(active_sessions),
            "max_allowed": MAX_SESSIONS
        },
        "config": {
            "text_weight": settings.TEXT_WEIGHT,
            "facial_weight": settings.FACIAL_WEIGHT,
            "incongruence_threshold": settings.INCONGRUENCE_THRESHOLD
        }
    }


# ============= CHAT ENDPOINTS =============

@app.post("/api/chat/message")
async def process_message(
    session_id: str = Form(...),
    message: str = Form(...),
    facial_frame: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Process a chat message with optional facial frame (AUTHENTICATED)
    
    Args:
        session_id: Unique session identifier
        message: User's text message
        facial_frame: Optional image file from webcam
        current_user: Authenticated user from token
    
    Returns:
        JSON with emotion analysis and therapeutic response
    """
    try:
        user_id = str(current_user["_id"])
        
        # Check session limit
        if len(active_sessions) >= MAX_SESSIONS and session_id not in active_sessions:
            raise HTTPException(status_code=503, detail="Server at capacity, please try again later")
        
        # Initialize session if new
        if session_id not in active_sessions:
            logger.info(f"New session started: {session_id} (User: {current_user['email']})")
            active_sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "history": [],
                "emotion_history": [],
                "message_count": 0
            }
        
        session = active_sessions[session_id]
        session["message_count"] += 1
        
        # Process facial frame if provided
        facial_emotion = None
        facial_confidence = 0.0
        
        if facial_frame and facial_model:
            try:
                # Read and decode image
                image_bytes = await facial_frame.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Get facial emotion
                    facial_emotion, facial_confidence = facial_model.predict(frame)
                    logger.debug(f"Facial emotion: {facial_emotion} ({facial_confidence:.2f})")
            except Exception as e:
                logger.error(f"Error processing facial frame: {e}")
        
        # Process text emotion
        text_emotion = None
        text_confidence = 0.0
        
        if text_model and message:
            try:
                text_emotion, text_confidence = text_model.predict(message)
                logger.debug(f"Text emotion: {text_emotion} ({text_confidence:.2f})")
            except Exception as e:
                logger.error(f"Error processing text: {e}")
        
        # Fuse predictions
        emotion_result = fusion_model.fuse(
            text_emotion=text_emotion,
            text_confidence=text_confidence,
            facial_emotion=facial_emotion,
            facial_confidence=facial_confidence
        )
        
        logger.info(f"Session {session_id} - Final emotion: {emotion_result['final_emotion']} "
                   f"({emotion_result['confidence']:.2f}) - Method: {emotion_result['fusion_method']}")
        
        # Check for crisis keywords
        is_crisis = llm_service.detect_crisis_keywords(message) if llm_service else False
        
        # Generate therapeutic response
        if llm_service and not is_crisis:
            try:
                response = await llm_service.generate_response(
                    user_message=message,
                    detected_emotion=emotion_result['final_emotion'],
                    emotional_context=emotion_result,
                    session_history=session['history']
                )
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                response = _get_fallback_response(emotion_result['final_emotion'])
        elif is_crisis and llm_service:
            response = await llm_service.generate_crisis_response()
            logger.warning(f"Crisis keywords detected in session {session_id}")
        else:
            response = _get_fallback_response(emotion_result['final_emotion'])
        
        # Update session history (in-memory)
        session['history'].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "bot_response": response,
            "emotion_data": emotion_result
        })
        
        session['emotion_history'].append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_result['final_emotion'],
            "confidence": emotion_result['confidence'],
            "text_emotion": text_emotion,
            "facial_emotion": facial_emotion,
            "incongruence": emotion_result.get('incongruence', False)
        })
        
        # Save to MongoDB for persistence
        try:
            sessions_collection = get_sessions_collection()
            await sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$setOnInsert": {
                        "user_id": user_id,
                        "started_at": datetime.utcnow()
                    },
                    "$inc": {"total_messages": 1},
                    "$push": {
                        "messages": {
                            "sender": "user",
                            "content": message,
                            "emotion": emotion_result['final_emotion'],
                            "confidence": emotion_result['confidence'],
                            "timestamp": datetime.utcnow()
                        }
                    },
                    "$set": {
                        "last_updated": datetime.utcnow(),
                        "last_emotion": emotion_result['final_emotion']
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
        
        # Trim in-memory history if too long (keep last 50 messages)
        if len(session['history']) > 50:
            session['history'] = session['history'][-50:]
        if len(session['emotion_history']) > 50:
            session['emotion_history'] = session['emotion_history'][-50:]
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "response": response,
            "emotion_analysis": {
                "detected_emotion": emotion_result['final_emotion'],
                "confidence": emotion_result['confidence'],
                "text_emotion": emotion_result.get('text_emotion'),
                "facial_emotion": emotion_result.get('facial_emotion'),
                "incongruence_detected": emotion_result.get('incongruence', False),
                "incongruence_message": emotion_result.get('incongruence_message'),
                "fusion_method": emotion_result['fusion_method']
            },
            "is_crisis": is_crisis,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with continuous camera feed
    NOTE: WebSocket authentication should be implemented separately
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")
    
    # Initialize session
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "created_at": datetime.now(),
            "history": [],
            "emotion_history": [],
            "websocket": websocket,
            "latest_facial_emotion": None,
            "latest_facial_confidence": 0.0
        }
    else:
        active_sessions[session_id]["websocket"] = websocket
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            message_type = data.get('type')
            
            if message_type == 'text_message':
                # Process text message
                message = data.get('message')
                
                if not message:
                    continue
                
                logger.debug(f"Received text message: {message[:50]}...")
                
                # Get text emotion
                text_emotion, text_conf = text_model.predict(message) if text_model else (None, 0.0)
                
                # Get latest facial emotion from continuous stream
                facial_emotion = active_sessions[session_id].get('latest_facial_emotion')
                facial_conf = active_sessions[session_id].get('latest_facial_confidence', 0.0)
                
                # Fuse predictions
                emotion_result = fusion_model.fuse(
                    text_emotion=text_emotion,
                    text_confidence=text_conf,
                    facial_emotion=facial_emotion,
                    facial_confidence=facial_conf
                )
                
                # Check for crisis
                is_crisis = llm_service.detect_crisis_keywords(message) if llm_service else False
                
                # Generate response
                if llm_service and not is_crisis:
                    try:
                        response = await llm_service.generate_response(
                            user_message=message,
                            detected_emotion=emotion_result['final_emotion'],
                            emotional_context=emotion_result,
                            session_history=active_sessions[session_id]['history']
                        )
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        response = _get_fallback_response(emotion_result['final_emotion'])
                elif is_crisis:
                    response = await llm_service.generate_crisis_response()
                else:
                    response = _get_fallback_response(emotion_result['final_emotion'])
                
                # Update history
                active_sessions[session_id]['history'].append({
                    "timestamp": datetime.now().isoformat(),
                    "user_message": message,
                    "bot_response": response,
                    "emotion_data": emotion_result
                })
                
                # Send response back
                await websocket.send_json({
                    "type": "bot_response",
                    "response": response,
                    "emotion_analysis": emotion_result,
                    "is_crisis": is_crisis,
                    "timestamp": datetime.now().isoformat()
                })
                
            elif message_type == 'facial_frame':
                # Process facial frame (continuous monitoring)
                frame_data = data.get('frame')  # Base64 encoded
                
                if not frame_data or not facial_model:
                    continue
                
                try:
                    # Decode frame
                    frame_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Get facial emotion
                        facial_emotion, facial_conf = facial_model.predict(frame)
                        
                        # Update session with latest facial emotion
                        active_sessions[session_id]['latest_facial_emotion'] = facial_emotion
                        active_sessions[session_id]['latest_facial_confidence'] = facial_conf
                        
                        # Send real-time emotion feedback
                        await websocket.send_json({
                            "type": "facial_emotion_update",
                            "emotion": facial_emotion,
                            "confidence": float(facial_conf),
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error processing facial frame: {e}")
            
            elif message_type == 'ping':
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        if session_id in active_sessions:
            active_sessions[session_id].pop('websocket', None)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id].pop('websocket', None)


@app.get("/api/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get conversation history for a session (AUTHENTICATED)"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Verify user owns this session
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to view this session")
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "message_count": len(session['history']),
        "history": session['history'],
        "emotion_history": session['emotion_history']
    }


@app.get("/api/session/{session_id}/analysis")
async def get_session_analysis(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get emotion analysis for a session (AUTHENTICATED)"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Verify user owns this session
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to view this session")
    
    emotion_history = session['emotion_history']
    
    if not emotion_history:
        return {"error": "No emotion data available for this session"}
    
    # Use fusion model to analyze session
    analysis = fusion_model.analyze_session_emotions(emotion_history)
    
    return {
        "session_id": session_id,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/api/session/{session_id}")
async def end_session(
    session_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """End a session and clear its data (AUTHENTICATED)"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Verify user owns this session
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to end this session")
    
    # Close WebSocket if open
    if 'websocket' in session:
        try:
            await session['websocket'].close()
        except:
            pass
    
    # Remove session
    del active_sessions[session_id]
    logger.info(f"Session ended: {session_id}")
    
    return {"message": "Session ended successfully", "session_id": session_id}


# ============= UTILITY FUNCTIONS =============

def _get_fallback_response(emotion: str) -> str:
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )