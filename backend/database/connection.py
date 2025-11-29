"""
MongoDB Database Connection
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB Connection Manager"""
    
    client: Optional[AsyncIOMotorClient] = None
    database = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        try:
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
            db_name = os.getenv("MONGODB_NAME", "mental_health_db")
            
            logger.info(f"Connecting to MongoDB: {mongodb_uri}")
            
            cls.client = AsyncIOMotorClient(mongodb_uri)
            cls.database = cls.client[db_name]
            
            # Test connection
            await cls.client.admin.command('ping')
            logger.info("✓ MongoDB connected successfully")
            
            # Create indexes
            await cls.create_indexes()
            
        except Exception as e:
            logger.error(f"✗ MongoDB connection error: {e}")
            raise
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")
    
    @classmethod
    async def create_indexes(cls):
        """Create database indexes"""
        try:
            await cls.database.users.create_index("email", unique=True)
            await cls.database.counselors.create_index("email", unique=True)
            await cls.database.bookings.create_index("user_id")
            await cls.database.bookings.create_index("counselor_id")
            await cls.database.contacts.create_index("email")
            
            logger.info("✓ Database indexes created")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    @classmethod
    def get_database(cls):
        """Get database instance"""
        return cls.database


# Collections helper
def get_users_collection():
    return MongoDB.database.users

def get_counselors_collection():
    return MongoDB.database.counselors

def get_bookings_collection():
    return MongoDB.database.bookings

def get_contacts_collection():
    return MongoDB.database.contacts

def get_sessions_collection():
    return MongoDB.database.chat_sessions