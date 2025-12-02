import React, { useState, useRef, useEffect } from 'react';
import Navbar from '../../components/shared/Navbar';
import { chatAPI } from '../../services/api';
import Webcam from 'react-webcam';
import EmotionBadge from '../../components/shared/EmotionBadge';
import {
  PaperAirplaneIcon,
  VideoCameraIcon,
  XMarkIcon,
  ChatBubbleLeftRightIcon,
  MicrophoneIcon,
} from '@heroicons/react/24/outline';
import type { ChatResponse } from '../../types';

interface Message {
  role: 'user' | 'bot';
  content: string;
  emotion?: string;
  confidence?: number;
  timestamp: Date;
}

// Extend Window interface for speech recognition
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(`session-${Date.now()}`);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [lastEmotion, setLastEmotion] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  
  const webcamRef = useRef<Webcam>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported in this browser');
      setSpeechSupported(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      console.log('Speech recognition started');
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      // Update input with final transcript
      if (finalTranscript) {
        setInput((prev) => prev + finalTranscript);
      }
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      
      if (event.error === 'not-allowed') {
        alert('Microphone access denied. Please enable microphone permissions.');
      } else if (event.error === 'no-speech') {
        console.log('No speech detected, restarting...');
        // Auto-restart if no speech detected
        if (isListening) {
          setTimeout(() => {
            try {
              recognition.start();
            } catch (e) {
              console.log('Recognition already started');
            }
          }, 100);
        }
      }
    };

    recognition.onend = () => {
      console.log('Speech recognition ended');
      setIsListening(false);
      
      // Auto-restart if still supposed to be listening
      if (recognitionRef.current && isListening) {
        try {
          recognition.start();
        } catch (e) {
          console.log('Could not restart recognition');
        }
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleSpeechRecognition = () => {
    if (!speechSupported) {
      alert('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
      return;
    }

    if (!recognitionRef.current) {
      console.error('Speech recognition not initialized');
      return;
    }

    if (isListening) {
      // Stop listening
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      // Start listening
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (error) {
        console.error('Error starting speech recognition:', error);
        // If already started, just update state
        setIsListening(true);
      }
    }
  };

  const captureFrame = async (): Promise<File | undefined> => {
    if (!cameraEnabled || !webcamRef.current) return undefined;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return undefined;

    const response = await fetch(imageSrc);
    const blob = await response.blob();
    return new File([blob], 'frame.jpg', { type: 'image/jpeg' });
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    // Stop speech recognition when sending
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const frame = await captureFrame();
      const response: ChatResponse = await chatAPI.sendMessage(
        sessionId,
        input,
        frame
      );

      const botMessage: Message = {
        role: 'bot',
        content: response.response,
        emotion: response.emotion_analysis.detected_emotion,
        confidence: response.emotion_analysis.confidence,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
      setLastEmotion(response.emotion_analysis.detected_emotion);

      // Show incongruence alert if detected
      if (response.emotion_analysis.incongruence_detected) {
        setTimeout(() => {
          alert(
            `⚠️ Emotional Incongruence Detected:\n\n${response.emotion_analysis.incongruence_message}`
          );
        }, 500);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'bot',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="bg-white rounded-2xl shadow-xl h-[calc(100vh-8rem)] flex flex-col relative">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary-600 to-primary-500 p-4 rounded-t-2xl flex-shrink-0">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-white">AI Therapy Chat</h2>
                <p className="text-primary-100 text-sm">
                  I'm here to listen and support you
                </p>
              </div>
              <div className="flex items-center gap-4">
                {lastEmotion && (
                  <div className="bg-white rounded-lg px-4 py-2">
                    <EmotionBadge emotion={lastEmotion} />
                  </div>
                )}
                <button
                  onClick={() => setCameraEnabled(!cameraEnabled)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    cameraEnabled
                      ? 'bg-red-500 text-white hover:bg-red-600'
                      : 'bg-white text-primary-600 hover:bg-primary-50'
                  }`}
                >
                  {cameraEnabled ? (
                    <>
                      <XMarkIcon className="h-5 w-5 inline mr-2" />
                      Disable Camera
                    </>
                  ) : (
                    <>
                      <VideoCameraIcon className="h-5 w-5 inline mr-2" />
                      Enable Camera
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-16">
                <ChatBubbleLeftRightIcon className="h-20 w-20 mx-auto text-gray-300 mb-4" />
                <h3 className="text-xl font-medium text-gray-900 mb-2">
                  Start a conversation
                </h3>
                <p className="text-gray-500 text-lg">
                  Share how you're feeling, and I'll provide support and guidance.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-2xl rounded-2xl px-6 py-4 ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="whitespace-pre-wrap text-base">{message.content}</p>
                  {message.emotion && (
                    <div className="mt-2 flex items-center gap-2">
                      <span className="text-xs opacity-75">Detected emotion:</span>
                      <EmotionBadge
                        emotion={message.emotion}
                        confidence={message.confidence}
                      />
                    </div>
                  )}
                  <p className="text-xs opacity-75 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl px-6 py-4">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.2s' }}
                    ></div>
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: '0.4s' }}
                    ></div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Camera Preview */}
          {cameraEnabled && (
            <div className="absolute bottom-40 right-6 z-10">
              <div className="relative rounded-lg overflow-hidden border-4 border-primary-500 shadow-2xl bg-black">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  className="w-48 h-36 object-cover"
                  videoConstraints={{
                    width: 640,
                    height: 480,
                    facingMode: 'user',
                  }}
                />
                <div className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded text-xs font-medium flex items-center">
                  <span className="animate-pulse mr-1">●</span> LIVE
                </div>
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                  <p className="text-white text-xs font-medium">Facial Emotion Detection Active</p>
                </div>
              </div>
            </div>
          )}

          {/* Input Area with Microphone */}
          <div className="border-t border-gray-200 p-4 bg-white rounded-b-2xl flex-shrink-0">
            {/* Listening Indicator */}
            {isListening && (
              <div className="mb-3 flex items-center gap-2 text-red-600 animate-pulse">
                <div className="flex space-x-1">
                  <div className="w-1 h-4 bg-red-600 rounded animate-pulse"></div>
                  <div className="w-1 h-6 bg-red-600 rounded animate-pulse" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-1 h-5 bg-red-600 rounded animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-1 h-7 bg-red-600 rounded animate-pulse" style={{ animationDelay: '0.3s' }}></div>
                  <div className="w-1 h-4 bg-red-600 rounded animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                </div>
                <span className="text-sm font-medium">Listening... (speak now)</span>
              </div>
            )}

            <div className="flex gap-4">
              {/* Microphone Button */}
              <button
                onClick={toggleSpeechRecognition}
                disabled={!speechSupported}
                className={`px-4 py-3 rounded-xl font-medium transition-all flex items-center gap-2 ${
                  isListening
                    ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
                title={speechSupported ? 'Click to start/stop voice input' : 'Speech recognition not supported'}
              >
                <MicrophoneIcon className="h-6 w-6" />
              </button>

              {/* Text Input */}
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message here or use the microphone... (Press Enter to send)"
                className="flex-1 resize-none border-2 border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-base"
                rows={3}
              />

              {/* Send Button */}
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="px-8 py-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 font-medium text-base"
              >
                <PaperAirplaneIcon className="h-5 w-5" />
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;