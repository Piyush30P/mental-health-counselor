// User Types
export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'user' | 'admin';
  created_at: string;
  is_active: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  full_name: string;
  password: string;
  role?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

// Counselor Types
export interface Counselor {
  id: string;
  name: string;
  specialization: string;
  email: string;
  phone: string;
  bio: string;
  availability: string[];
  image_url?: string;
  created_at: string;
  is_active: boolean;
}

export interface CounselorCreate {
  name: string;
  specialization: string;
  email: string;
  phone: string;
  bio: string;
  availability: string[];
  image_url?: string;
}

// Booking Types
export interface Booking {
  id: string;
  counselor_id: string;
  counselor_name: string;
  user_name: string;
  user_email: string;
  date: string;
  time_slot: string;
  reason: string;
  notes?: string;
  status: 'pending' | 'confirmed' | 'completed' | 'cancelled';
  created_at: string;
}

export interface BookingCreate {
  counselor_id: string;
  date: string;
  time_slot: string;
  reason: string;
  notes?: string;
}

// Contact Types
export interface Contact {
  id: string;
  name: string;
  email: string;
  subject: string;
  message: string;
  status: 'new' | 'replied' | 'closed';
  created_at: string;
}

export interface ContactCreate {
  name: string;
  email: string;
  subject: string;
  message: string;
}

// Chat Types
export interface EmotionAnalysis {
  detected_emotion: string;
  confidence: number;
  text_emotion: string;
  facial_emotion?: string;
  incongruence_detected: boolean;
  incongruence_message?: string;
  fusion_method: string;
}

export interface ChatMessage {
  session_id: string;
  message: string;
  facial_frame?: File;
}

export interface ChatResponse {
  success: boolean;
  session_id: string;
  response: string;
  emotion_analysis: EmotionAnalysis;
  is_crisis: boolean;
  timestamp: string;
}

export interface ChatSession {
  id: string;
  session_id: string;
  started_at: string;
  total_messages: number;
  emotion_summary: Record<string, number>;
}

// Admin Analytics
export interface Analytics {
  total_users: number;
  total_sessions: number;
  total_bookings: number;
  pending_bookings: number;
  active_users: number;
}