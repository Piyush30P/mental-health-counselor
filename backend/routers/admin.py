"""
Admin Router - Admin-specific endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database.models import (
    CounselorCreate, CounselorResponse,
    BookingResponse, ContactResponse
)
from database.connection import (
    get_counselors_collection, get_bookings_collection,
    get_contacts_collection, get_sessions_collection, get_users_collection
)
from auth.dependencies import get_current_admin_user
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ============= DASHBOARD ANALYTICS =============

@router.get("/analytics")
async def get_analytics(current_user: dict = Depends(get_current_admin_user)):
    """Get dashboard analytics"""
    
    users_collection = get_users_collection()
    sessions_collection = get_sessions_collection()
    bookings_collection = get_bookings_collection()
    
    total_users = await users_collection.count_documents({"role": "user"})
    total_sessions = await sessions_collection.count_documents({})
    total_bookings = await bookings_collection.count_documents({})
    pending_bookings = await bookings_collection.count_documents({"status": "pending"})
    
    return {
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_bookings": total_bookings,
        "pending_bookings": pending_bookings,
        "active_users": total_users  # Can be refined with last_active logic
    }


# ============= COUNSELOR MANAGEMENT =============

@router.post("/counselors", response_model=CounselorResponse)
async def add_counselor(
    counselor_data: CounselorCreate,
    current_user: dict = Depends(get_current_admin_user)
):
    """Add new counselor"""
    
    counselors_collection = get_counselors_collection()
    
    # Check if email exists
    existing = await counselors_collection.find_one({"email": counselor_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Counselor email already exists")
    
    counselor_dict = {
        "name": counselor_data.name,
        "specialization": counselor_data.specialization,
        "email": counselor_data.email,
        "phone": counselor_data.phone,
        "bio": counselor_data.bio,
        "availability": counselor_data.availability,
        "image_url": counselor_data.image_url,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    result = await counselors_collection.insert_one(counselor_dict)
    
    return CounselorResponse(
        id=str(result.inserted_id),
        **counselor_dict
    )


@router.get("/counselors", response_model=List[CounselorResponse])
async def get_all_counselors(current_user: dict = Depends(get_current_admin_user)):
    """Get all counselors (including inactive)"""
    
    counselors_collection = get_counselors_collection()
    counselors = await counselors_collection.find({}).to_list(100)
    
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


@router.delete("/counselors/{counselor_id}")
async def delete_counselor(
    counselor_id: str,
    current_user: dict = Depends(get_current_admin_user)
):
    """Delete/deactivate counselor"""
    
    counselors_collection = get_counselors_collection()
    result = await counselors_collection.update_one(
        {"_id": ObjectId(counselor_id)},
        {"$set": {"is_active": False}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Counselor not found")
    
    return {"message": "Counselor deactivated successfully"}


# ============= BOOKING MANAGEMENT =============

@router.get("/bookings", response_model=List[BookingResponse])
async def get_all_bookings(current_user: dict = Depends(get_current_admin_user)):
    """Get all bookings"""
    
    bookings_collection = get_bookings_collection()
    bookings = await bookings_collection.find({}).sort("created_at", -1).to_list(200)
    
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


@router.patch("/bookings/{booking_id}/status")
async def update_booking_status(
    booking_id: str,
    status: str,
    current_user: dict = Depends(get_current_admin_user)
):
    """Update booking status"""
    
    if status not in ["pending", "confirmed", "completed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    bookings_collection = get_bookings_collection()
    result = await bookings_collection.update_one(
        {"_id": ObjectId(booking_id)},
        {"$set": {"status": status}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return {"message": "Booking status updated"}


# ============= SESSION MANAGEMENT =============

@router.get("/sessions")
async def get_all_sessions(
    limit: int = 50,
    current_user: dict = Depends(get_current_admin_user)
):
    """Get all chat sessions"""
    
    sessions_collection = get_sessions_collection()
    sessions = await sessions_collection.find({}).sort("started_at", -1).limit(limit).to_list(limit)
    
    return [
        {
            "id": str(s["_id"]),
            "user_id": s.get("user_id"),
            "session_id": s.get("session_id"),
            "started_at": s.get("started_at"),
            "total_messages": s.get("total_messages", 0),
            "emotion_summary": s.get("emotion_summary", {}),
            "incongruence_count": s.get("incongruence_count", 0)
        }
        for s in sessions
    ]


# ============= CONTACT MANAGEMENT =============

@router.get("/contacts", response_model=List[ContactResponse])
async def get_all_contacts(current_user: dict = Depends(get_current_admin_user)):
    """Get all contact submissions"""
    
    contacts_collection = get_contacts_collection()
    contacts = await contacts_collection.find({}).sort("created_at", -1).to_list(100)
    
    return [
        ContactResponse(
            id=str(c["_id"]),
            name=c["name"],
            email=c["email"],
            subject=c["subject"],
            message=c["message"],
            status=c["status"],
            created_at=c["created_at"]
        )
        for c in contacts
    ]