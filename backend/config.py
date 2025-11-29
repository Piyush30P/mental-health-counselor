"""
Configuration Management
Loads and validates environment variables
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    
    # Model Paths
    FACIAL_MODEL_PATH: str = "weights/facial_model.h5"
    TEXT_MODEL_PATH: str = "weights/text_model.h5"
    VOCAB_PATH: str = "weights/word2idx.pkl"
    CONFIG_PATH: str = "weights/config.json"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # Fusion Weights
    TEXT_WEIGHT: float = 0.6
    FACIAL_WEIGHT: float = 0.4
    
    # Incongruence Detection
    INCONGRUENCE_THRESHOLD: float = 0.7
    
    # LLM Configuration
    LLM_MODEL: str = "gpt-4-turbo-preview"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 200
    
    # Session Configuration
    SESSION_TIMEOUT: int = 3600
    MAX_SESSIONS: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # ============= NEW: DATABASE CONFIGURATION =============
    
    # MongoDB Settings
    MONGODB_URI: str = "mongodb://127.0.0.1:27017"
    MONGODB_NAME: str = "mental_health_db"
    
    # ============= NEW: AUTHENTICATION CONFIGURATION =============
    
    # JWT Settings
    JWT_SECRET: str = "your-super-secret-key-change-this-in-production-use-openssl-rand-hex-32"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    def get_allowed_origins(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    def validate_weights(self) -> bool:
        """Ensure fusion weights sum to 1.0"""
        total = self.TEXT_WEIGHT + self.FACIAL_WEIGHT
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total} "
                f"(TEXT_WEIGHT={self.TEXT_WEIGHT}, FACIAL_WEIGHT={self.FACIAL_WEIGHT})"
            )
        return True
    
    def validate_jwt_secret(self) -> bool:
        """Ensure JWT secret is secure in production"""
        if not self.DEBUG and self.JWT_SECRET == "your-super-secret-key-change-this-in-production-use-openssl-rand-hex-32":
            raise ValueError(
                "⚠️  SECURITY WARNING: You must change JWT_SECRET in production! "
                "Generate a secure key with: openssl rand -hex 32"
            )
        return True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env


# Create global settings instance
settings = Settings()

# Validate settings
try:
    settings.validate_weights()
    settings.validate_jwt_secret()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
    print("Please check your .env file")

# Display loaded configuration (without sensitive data)
if settings.DEBUG:
    print("\n" + "="*60)
    print("CONFIGURATION LOADED")
    print("="*60)
    print(f"OpenAI API Key: {'✓ Set' if settings.OPENAI_API_KEY else '✗ Missing'}")
    print(f"Facial Model: {settings.FACIAL_MODEL_PATH}")
    print(f"Text Model: {settings.TEXT_MODEL_PATH}")
    print(f"Server: {settings.HOST}:{settings.PORT}")
    print(f"Fusion Weights: Text={settings.TEXT_WEIGHT}, Facial={settings.FACIAL_WEIGHT}")
    print(f"Allowed Origins: {len(settings.get_allowed_origins())} configured")
    print(f"MongoDB: {settings.MONGODB_URI} → {settings.MONGODB_NAME}")
    print(f"JWT Secret: {'✓ Set' if settings.JWT_SECRET else '✗ Missing'}")
    print(f"Token Expiry: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print("="*60 + "\n")