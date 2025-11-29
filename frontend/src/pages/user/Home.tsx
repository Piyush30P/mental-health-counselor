import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { userAPI } from '../../services/api';
import Navbar from '../../components/shared/Navbar';
import {
  ChatBubbleLeftRightIcon,
  CalendarIcon,
  HeartIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import type { Booking, ChatSession } from '../../types';

const Home: React.FC = () => {
  const { user } = useAuth();
  const [recentBookings, setRecentBookings] = useState<Booking[]>([]);
  const [recentSessions, setRecentSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [bookings, sessions] = await Promise.all([
        userAPI.getMyBookings(),
        userAPI.getMySessions(5),
      ]);
      setRecentBookings(bookings.slice(0, 3));
      setRecentSessions(sessions);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      title: 'AI Therapy Chat',
      description: 'Talk to our AI counselor with real-time emotion detection',
      icon: ChatBubbleLeftRightIcon,
      link: '/user/chat',
      color: 'from-blue-500 to-blue-600',
    },
    {
      title: 'Book Counselor',
      description: 'Schedule appointments with professional therapists',
      icon: CalendarIcon,
      link: '/user/book',
      color: 'from-green-500 to-green-600',
    },
    {
      title: 'Wellness Resources',
      description: 'Access mental health resources and support',
      icon: HeartIcon,
      link: '/user/contact',
      color: 'from-pink-500 to-pink-600',
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="bg-gradient-to-r from-primary-600 to-primary-500 rounded-2xl shadow-xl p-8 mb-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">
                Welcome back, {user?.full_name}! 👋
              </h1>
              <p className="text-primary-100 text-lg">
                How are you feeling today? I'm here to support you.
              </p>
            </div>
            <SparklesIcon className="h-20 w-20 opacity-50" />
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Link
                key={feature.title}
                to={feature.link}
                className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all p-6 group"
              >
                <div
                  className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
                >
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </Link>
            );
          })}
        </div>

        {/* Recent Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Bookings */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">Recent Bookings</h2>
              <Link
                to="/user/book"
                className="text-primary-600 hover:text-primary-700 text-sm font-medium"
              >
                View All
              </Link>
            </div>
            {loading ? (
              <div className="animate-pulse space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-20 bg-gray-200 rounded"></div>
                ))}
              </div>
            ) : recentBookings.length > 0 ? (
              <div className="space-y-3">
                {recentBookings.map((booking) => (
                  <div
                    key={booking.id}
                    className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">
                          {booking.counselor_name}
                        </h3>
                        <p className="text-sm text-gray-500">
                          {booking.date} at {booking.time_slot}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${
                          booking.status === 'confirmed'
                            ? 'bg-green-100 text-green-800'
                            : booking.status === 'pending'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {booking.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <CalendarIcon className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                <p>No bookings yet</p>
                <Link
                  to="/user/book"
                  className="text-primary-600 hover:text-primary-700 text-sm font-medium mt-2 inline-block"
                >
                  Book your first session
                </Link>
              </div>
            )}
          </div>

          {/* Recent Chat Sessions */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">Chat History</h2>
              <Link
                to="/user/chat"
                className="text-primary-600 hover:text-primary-700 text-sm font-medium"
              >
                Start Chat
              </Link>
            </div>
            {loading ? (
              <div className="animate-pulse space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-20 bg-gray-200 rounded"></div>
                ))}
              </div>
            ) : recentSessions.length > 0 ? (
              <div className="space-y-3">
                {recentSessions.map((session) => (
                  <div
                    key={session.id}
                    className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">
                          Session {session.session_id.split('-')[1]}
                        </h3>
                        <p className="text-sm text-gray-500">
                          {new Date(session.started_at).toLocaleDateString()} -{' '}
                          {session.total_messages} messages
                        </p>
                      </div>
                      {session.emotion_summary && (
                        <div className="flex gap-1">
                          {Object.entries(session.emotion_summary)
                            .slice(0, 3)
                            .map(([emotion, count]) => (
                              <span
                                key={emotion}
                                className="px-2 py-1 bg-primary-100 text-primary-800 text-xs rounded-full"
                              >
                                {emotion}: {count}
                              </span>
                            ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <ChatBubbleLeftRightIcon className="h-12 w-12 mx-auto mb-2 text-gray-400" />
                <p>No chat sessions yet</p>
                <Link
                  to="/user/chat"
                  className="text-primary-600 hover:text-primary-700 text-sm font-medium mt-2 inline-block"
                >
                  Start your first conversation
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Crisis Resources */}
        <div className="bg-red-50 border-l-4 border-red-400 rounded-lg p-6 mt-8">
          <div className="flex">
            <div className="flex-shrink-0">
              <HeartIcon className="h-6 w-6 text-red-400" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Need immediate help?
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>🆘 National Suicide Prevention Lifeline: 988 or 1-800-273-8255</p>
                <p>🆘 Crisis Text Line: Text 'HELLO' to 741741</p>
                <p className="mt-1">Available 24/7 - You are not alone.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;