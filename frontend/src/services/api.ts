import axios, { AxiosError } from 'axios';
import type {
  AuthResponse,
  LoginRequest,
  RegisterRequest,
  User,
  Counselor,
  CounselorCreate,
  Booking,
  BookingCreate,
  Contact,
  ContactCreate,
  ChatResponse,
  ChatSession,
  Analytics,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============= AUTH API =============
export const authAPI = {
  register: async (data: RegisterRequest): Promise<User> => {
    const response = await api.post<User>('/api/auth/register', data);
    return response.data;
  },

  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/api/auth/login', data);
    return response.data;
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>('/api/auth/me');
    return response.data;
  },
};

// ============= USER API =============
export const userAPI = {
  getCounselors: async (): Promise<Counselor[]> => {
    const response = await api.get<Counselor[]>('/api/user/counselors');
    return response.data;
  },

  createBooking: async (data: BookingCreate): Promise<Booking> => {
    const response = await api.post<Booking>('/api/user/bookings', data);
    return response.data;
  },

  getMyBookings: async (): Promise<Booking[]> => {
    const response = await api.get<Booking[]>('/api/user/bookings');
    return response.data;
  },

  submitContact: async (data: ContactCreate): Promise<Contact> => {
    const response = await api.post<Contact>('/api/user/contact', data);
    return response.data;
  },

  getMySessions: async (limit = 10): Promise<ChatSession[]> => {
    const response = await api.get<ChatSession[]>('/api/user/sessions', {
      params: { limit },
    });
    return response.data;
  },
};

// ============= ADMIN API =============
export const adminAPI = {
  getAnalytics: async (): Promise<Analytics> => {
    const response = await api.get<Analytics>('/api/admin/analytics');
    return response.data;
  },

  addCounselor: async (data: CounselorCreate): Promise<Counselor> => {
    const response = await api.post<Counselor>('/api/admin/counselors', data);
    return response.data;
  },

  getAllCounselors: async (): Promise<Counselor[]> => {
    const response = await api.get<Counselor[]>('/api/admin/counselors');
    return response.data;
  },

  deleteCounselor: async (id: string): Promise<void> => {
    await api.delete(`/api/admin/counselors/${id}`);
  },

  getAllBookings: async (): Promise<Booking[]> => {
    const response = await api.get<Booking[]>('/api/admin/bookings');
    return response.data;
  },

  updateBookingStatus: async (
    id: string,
    status: string
  ): Promise<void> => {
    await api.patch(`/api/admin/bookings/${id}/status?status=${status}`);
  },

  getAllSessions: async (limit = 50): Promise<any[]> => {
    const response = await api.get('/api/admin/sessions', {
      params: { limit },
    });
    return response.data;
  },

  getAllContacts: async (): Promise<Contact[]> => {
    const response = await api.get<Contact[]>('/api/admin/contacts');
    return response.data;
  },
};

// ============= CHAT API =============
export const chatAPI = {
  sendMessage: async (
    sessionId: string,
    message: string,
    facialFrame?: File
  ): Promise<ChatResponse> => {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('message', message);
    if (facialFrame) {
      formData.append('facial_frame', facialFrame);
    }

    const response = await api.post<ChatResponse>('/api/chat/message', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getSessionHistory: async (sessionId: string): Promise<any> => {
    const response = await api.get(`/api/session/${sessionId}/history`);
    return response.data;
  },

  getSessionAnalysis: async (sessionId: string): Promise<any> => {
    const response = await api.get(`/api/session/${sessionId}/analysis`);
    return response.data;
  },

  endSession: async (sessionId: string): Promise<void> => {
    await api.delete(`/api/session/${sessionId}`);
  },
};

export default api;