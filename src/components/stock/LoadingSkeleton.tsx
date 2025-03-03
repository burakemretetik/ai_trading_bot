
import React from 'react';

const LoadingSkeleton: React.FC = () => {
  return (
    <div className="animate-pulse space-y-4">
      {[...Array(10)].map((_, i) => (
        <div key={i} className="h-20 bg-muted rounded-lg"></div>
      ))}
    </div>
  );
};

export default LoadingSkeleton;
