
import React from 'react';
import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface RefreshNewsButtonProps {
  onRefresh: () => Promise<void>;
  loading: boolean;
}

const RefreshNewsButton: React.FC<RefreshNewsButtonProps> = ({ onRefresh, loading }) => {
  return (
    <Button 
      variant="outline" 
      size="sm" 
      onClick={onRefresh} 
      disabled={loading}
      className="w-full"
    >
      {loading ? (
        <RefreshCw className="h-4 w-4 animate-spin" />
      ) : (
        <>
          <RefreshCw className="h-4 w-4 mr-2" />
          <span>Refresh News</span>
        </>
      )}
    </Button>
  );
};

export default RefreshNewsButton;
