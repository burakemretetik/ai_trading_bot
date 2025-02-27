
import React from 'react';
import { SignalStrength } from '@/utils/types';

interface SignalIndicatorProps {
  strength: SignalStrength;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

const SignalIndicator: React.FC<SignalIndicatorProps> = ({ 
  strength, 
  size = 'md', 
  showLabel = false,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'h-2 w-2',
    md: 'h-3 w-3',
    lg: 'h-4 w-4'
  };
  
  const colorClasses = {
    strong: 'bg-signal-strong',
    medium: 'bg-signal-medium',
    weak: 'bg-signal-weak',
    neutral: 'bg-signal-neutral'
  };
  
  const labels = {
    strong: 'Strong Buy Signal',
    medium: 'Moderate Signal',
    weak: 'Negative Signal',
    neutral: 'Neutral'
  };
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className={`rounded-full ${sizeClasses[size]} ${colorClasses[strength]} signal-pulse`}></span>
      {showLabel && (
        <span className="text-xs font-medium">{labels[strength]}</span>
      )}
    </div>
  );
};

export default SignalIndicator;
