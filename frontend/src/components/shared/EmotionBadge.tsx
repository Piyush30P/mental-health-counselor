import React from 'react';

interface EmotionBadgeProps {
  emotion: string;
  confidence?: number;
}

const EmotionBadge: React.FC<EmotionBadgeProps> = ({ emotion, confidence }) => {
  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      Happy: 'bg-green-100 text-green-800',
      Sad: 'bg-blue-100 text-blue-800',
      Angry: 'bg-red-100 text-red-800',
      Fearful: 'bg-purple-100 text-purple-800',
      Surprised: 'bg-yellow-100 text-yellow-800',
      Neutral: 'bg-gray-100 text-gray-800',
      Disgusted: 'bg-orange-100 text-orange-800',
    };
    return colors[emotion] || 'bg-gray-100 text-gray-800';
  };

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getEmotionColor(
        emotion
      )}`}
    >
      {emotion}
      {confidence && ` (${(confidence * 100).toFixed(0)}%)`}
    </span>
  );
};

export default EmotionBadge;