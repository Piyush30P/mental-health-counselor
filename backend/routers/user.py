"""
User Router - User-specific endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database.models import (
    CounselorResponse, BookingCreate, BookingResponse, 
    ContactCreate, ContactResponse
)
from database.connection import (
    get_counselors_collection, get_bookings_collection,
    get_contacts_collection, get_sessions_collection
)
from auth.dependencies import get_current_active_user
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/api/user", tags=["User"])


# ============= COUNSELORS =============

@router.get("/counselors", response_model=List[CounselorResponse])
async def get_counselors(current_user: dict = Depends(get_current_active_user)):
    """Get all active counselors"""
    
    counselors_collection = get_counselors_collection()
    counselors = await counselors_collection.find({"is_active": True}).to_list(100)
    
    return [
        CounselorResponse(
            id=str(c["_id"]),
            name=c["name"],
            specialization=c["specialization"],
            email=c["email"],
            phone=c["phone"],
            bio=c["bio"],
            availability=c.get("availability", []),
            image_url=c.get("image_url"),
            created_at=c["created_at"],
            is_active=c["is_active"]
        )
        for c in counselors
    ]


# ============= BOOKINGS =============

@router.post("/bookings", response_model=BookingResponse)
async def create_booking(
    booking_data: BookingCreate,
    current_user: dict = Depends(get_current_active_user)
):
    """Create new booking"""
    
    counselors_collection = get_counselors_collection()
    bookings_collection = get_bookings_collection()
    
    # Get counselor
    counselor = await counselors_collection.find_one(
        {"_id": ObjectId(booking_data.counselor_id)}
    )
    
    if not counselor:
        raise HTTPException(status_code=404, detail="Counselor not found")
    
    # Create booking
    booking_dict = {
        "counselor_id": booking_data.counselor_id,
        "counselor_name": counselor["name"],
        "user_id": str(current_user["_id"]),
        "user_name": current_user["full_name"],
        "user_email": current_user["email"],
        "date": booking_data.date,
        "time_slot": booking_data.time_slot,
        "reason": booking_data.reason,
        "notes": booking_data.notes,
        "status": "pending",
        "created_at": datetime.utcnow()
    }
    
    result = await bookings_collection.insert_one(booking_dict)
    
    return BookingResponse(
        id=str(result.inserted_id),
        **booking_dict
    )


@router.get("/bookings", response_model=List[BookingResponse])
async def get_my_bookings(current_user: dict = Depends(get_current_active_user)):
    """Get user's bookings"""
    
    bookings_collection = get_bookings_collection()
    bookings = await bookings_collection.find(
        {"user_id": str(current_user["_id"])}
    ).sort("created_at", -1).to_list(100)
    
    return [
        BookingResponse(
            id=str(b["_id"]),
            counselor_id=b["counselor_id"],
            counselor_name=b["counselor_name"],
            user_name=b["user_name"],
            user_email=b["user_email"],
            date=b["date"],
            time_slot=b["time_slot"],
            reason=b["reason"],
            notes=b.get("notes"),
            status=b["status"],
            created_at=b["created_at"]
        )
        for b in bookings
    ]


# ============= CONTACT =============

@router.post("/contact", response_model=ContactResponse)
async def submit_contact(
    contact_data: ContactCreate,
    current_user: dict = Depends(get_current_active_user)
):
    """Submit contact form"""
    
    contacts_collection = get_contacts_collection()
    
    contact_dict = {
        "name": contact_data.name,
        "email": contact_data.email,
        "subject": contact_data.subject,
        "message": contact_data.message,
        "status": "new",
        "created_at": datetime.utcnow()
    }
    
    result = await contacts_collection.insert_one(contact_dict)
    
    return ContactResponse(
        id=str(result.inserted_id),
        **contact_dict
    )


# ============= CHAT SESSIONS =============

@router.get("/sessions")
async def get_my_sessions(
    limit: int = 10,
    current_user: dict = Depends(get_current_active_user)
):
    """Get user's chat sessions"""
    
    sessions_collection = get_sessions_collection()
    sessions = await sessions_collection.find(
        {"user_id": str(current_user["_id"])}
    ).sort("started_at", -1).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(s["_id"]),
            "session_id": s.get("session_id"),
            "started_at": s.get("started_at"),
            "total_messages": s.get("total_messages", 0),
            "emotion_summary": s.get("emotion_summary", {})
        }
        for s in sessions
    ]