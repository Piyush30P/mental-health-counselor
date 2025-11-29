"""
MongoDB Models for Mental Health Platform - Pydantic v2 Compatible
"""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, EmailStr, Field, field_validator
from bson import ObjectId


# ============= OBJECT ID HANDLING =============

class PyObjectId(str):
    """Custom ObjectId type for Pydantic v2"""
    
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler):
        from pydantic_core import core_schema
        
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


# ============= USER MODELS =============

class UserBase(BaseModel):
    """Base User Model"""
    email: EmailStr
    full_name: str
    role: str = "user"  # user or admin


class UserCreate(UserBase):
    """User Registration Model"""
    password: str


class UserInDB(UserBase):
    """User in Database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserResponse(UserBase):
    """User Response Model"""
    id: str
    created_at: datetime
    is_active: bool


class LoginRequest(BaseModel):
    """Login Request"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token Response"""
    access_token: str
    token_type: str
    user: UserResponse


# ============= COUNSELOR MODELS =============

class CounselorBase(BaseModel):
    """Base Counselor Model"""
    name: str
    specialization: str
    email: EmailStr
    phone: str
    bio: str
    availability: List[str] = []
    image_url: Optional[str] = None


class CounselorCreate(CounselorBase):
    """Create Counselor"""
    pass


class CounselorInDB(CounselorBase):
    """Counselor in Database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CounselorResponse(CounselorBase):
    """Counselor Response"""
    id: str
    created_at: datetime
    is_active: bool


# ============= BOOKING MODELS =============

class BookingCreate(BaseModel):
    """Create Booking"""
    counselor_id: str
    date: str  # YYYY-MM-DD
    time_slot: str  # "10:00-11:00"
    reason: str
    notes: Optional[str] = None


class BookingInDB(BookingCreate):
    """Booking in Database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: str
    user_name: str
    user_email: str
    counselor_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, confirmed, completed, cancelled
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class BookingResponse(BaseModel):
    """Booking Response"""
    id: str
    counselor_id: str
    counselor_name: str
    user_name: str
    user_email: str
    date: str
    time_slot: str
    reason: str
    notes: Optional[str]
    status: str
    created_at: datetime


# ============= CONTACT MODELS =============

class ContactCreate(BaseModel):
    """Contact Form"""
    name: str
    email: EmailStr
    subject: str
    message: str


class ContactInDB(ContactCreate):
    """Contact in Database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "new"  # new, replied, closed
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ContactResponse(ContactCreate):
    """Contact Response"""
    id: str
    status: str
    created_at: datetime


# ============= CHAT SESSION MODELS =============

class MessageBase(BaseModel):
    """Single Chat Message"""
    sender: str  # "user" or "bot"
    content: str
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSessionBase(BaseModel):
    """Base Chat Session Model"""
    user_id: str
    session_id: str
    messages: List[MessageBase] = []


class ChatSessionInDB(ChatSessionBase):
    """Chat Session in Database"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: Optional[datetime] = None
    total_messages: int = 0
    last_emotion: Optional[str] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ChatSessionResponse(BaseModel):
    """Chat Session Response"""
    id: str
    user_id: str
    session_id: str
    started_at: datetime
    total_messages: int
    last_emotion: Optional[str]