
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
      variant="ghost" 
      size="sm" 
      onClick={onRefresh} 
      disabled={loading}
      className="w-full text-xs text-muted-foreground hover:text-foreground"
    >
      {loading ? (
        <RefreshCw className="h-3 w-3 animate-spin" />
      ) : (
        <>
          <RefreshCw className="h-3 w-3 mr-1" />
          <span>Refresh</span>
        </>
      )}
    </Button>
  );
};

export default RefreshNewsButton;
