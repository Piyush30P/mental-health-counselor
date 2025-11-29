import React, { useEffect, useState } from 'react';
import Navbar from '../../components/shared/Navbar';
import { userAPI } from '../../services/api';
import type { Counselor, BookingCreate } from '../../types';
import {
  CalendarIcon,
  ClockIcon,
  EnvelopeIcon,
  PhoneIcon,
} from '@heroicons/react/24/outline';

const BookCounselor: React.FC = () => {
  const [counselors, setCounselors] = useState<Counselor[]>([]);
  const [selectedCounselor, setSelectedCounselor] = useState<Counselor | null>(
    null
  );
  const [formData, setFormData] = useState<BookingCreate>({
    counselor_id: '',
    date: '',
    time_slot: '',
    reason: '',
    notes: '',
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    loadCounselors();
  }, []);

  const loadCounselors = async () => {
    try {
      const data = await userAPI.getCounselors();
      setCounselors(data);
    } catch (error) {
      console.error('Error loading counselors:', error);
    }
  };

  const handleSelectCounselor = (counselor: Counselor) => {
    setSelectedCounselor(counselor);
    setFormData({ ...formData, counselor_id: counselor.id });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await userAPI.createBooking(formData);
      setSuccess(true);
      setTimeout(() => {
        setSuccess(false);
        setSelectedCounselor(null);
        setFormData({
          counselor_id: '',
          date: '',
          time_slot: '',
          reason: '',
          notes: '',
        });
      }, 3000);
    } catch (error) {
      console.error('Error creating booking:', error);
      alert('Failed to create booking. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const timeSlots = [
    '09:00-10:00',
    '10:00-11:00',
    '11:00-12:00',
    '14:00-15:00',
    '15:00-16:00',
    '16:00-17:00',
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Book a Counselor</h1>
          <p className="text-gray-600 mt-2">
            Choose a professional therapist and schedule your session
          </p>
        </div>

        {success && (
          <div className="mb-6 bg-green-50 border border-green-200 text-green-800 px-6 py-4 rounded-lg">
            ✓ Booking created successfully! You will receive a confirmation email
            shortly.
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Counselors List */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">
                Available Counselors
              </h2>
              <div className="space-y-4">
                {counselors.map((counselor) => (
                  <div
                    key={counselor.id}
                    className={`border-2 rounded-lg p-6 cursor-pointer transition-all ${
                      selectedCounselor?.id === counselor.id
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-primary-300'
                    }`}
                    onClick={() => handleSelectCounselor(counselor)}
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-16 h-16 bg-gradient-to-br from-primary-400 to-primary-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                        {counselor.name.charAt(0)}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900">
                          {counselor.name}
                        </h3>
                        <p className="text-primary-600 text-sm font-medium mb-2">
                          {counselor.specialization}
                        </p>
                        <p className="text-gray-600 text-sm mb-3">
                          {counselor.bio}
                        </p>
                        <div className="flex flex-wrap gap-4 text-sm text-gray-500">
                          <div className="flex items-center">
                            <EnvelopeIcon className="h-4 w-4 mr-1" />
                            {counselor.email}
                          </div>
                          <div className="flex items-center">
                            <PhoneIcon className="h-4 w-4 mr-1" />
                            {counselor.phone}
                          </div>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {counselor.availability.map((day) => (
                            <span
                              key={day}
                              className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full"
                            >
                              {day}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Booking Form */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-md p-6 sticky top-8">
              <h2 className="text-xl font-bold text-gray-900 mb-6">
                Book Appointment
              </h2>

              {!selectedCounselor ? (
                <div className="text-center py-8 text-gray-500">
                  <CalendarIcon className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>Please select a counselor to continue</p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Selected Counselor
                    </label>
                    <div className="px-4 py-3 bg-primary-50 rounded-lg">
                      <p className="font-medium text-primary-900">
                        {selectedCounselor.name}
                      </p>
                      <p className="text-sm text-primary-600">
                        {selectedCounselor.specialization}
                      </p>
                    </div>
                  </div>

                  <div>
                    <label
                      htmlFor="date"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Date
                    </label>
                    <input
                      type="date"
                      id="date"
                      required
                      min={new Date().toISOString().split('T')[0]}
                      value={formData.date}
                      onChange={(e) =>
                        setFormData({ ...formData, date: e.target.value })
                      }
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="time_slot"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Time Slot
                    </label>
                    <select
                      id="time_slot"
                      required
                      value={formData.time_slot}
                      onChange={(e) =>
                        setFormData({ ...formData, time_slot: e.target.value })
                      }
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    >
                      <option value="">Select a time</option>
                      {timeSlots.map((slot) => (
                        <option key={slot} value={slot}>
                          {slot}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label
                      htmlFor="reason"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Reason for Appointment
                    </label>
                    <textarea
                      id="reason"
                      required
                      rows={3}
                      value={formData.reason}
                      onChange={(e) =>
                        setFormData({ ...formData, reason: e.target.value })
                      }
                      placeholder="Brief description of what you'd like to discuss..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="notes"
                      className="block text-sm font-medium text-gray-700 mb-2"
                    >
                      Additional Notes (Optional)
                    </label>
                    <textarea
                      id="notes"
                      rows={2}
                      value={formData.notes}
                      onChange={(e) =>
                        setFormData({ ...formData, notes: e.target.value })
                      }
                      placeholder="Any additional information..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Booking...' : 'Confirm Booking'}
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookCounselor;