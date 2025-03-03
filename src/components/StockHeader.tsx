
import React from 'react';
import { Star } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface StockHeaderProps {
  symbol: string;
  name: string;
  tracked: boolean;
  onToggleTracking: (id: string) => void;
  stockId: string;
}

const StockHeader: React.FC<StockHeaderProps> = ({
  symbol,
  name,
  tracked,
  onToggleTracking,
  stockId
}) => {
  return (
    <div className="flex items-center justify-between">
      <div className="flex flex-col">
        <h3 className="font-medium text-lg">{symbol}</h3>
        <p className="text-xs text-muted-foreground">{name}</p>
      </div>
      
      <Button 
        variant="ghost" 
        size="icon" 
        className="h-8 w-8 rounded-full" 
        onClick={() => onToggleTracking(stockId)}
      >
        <Star className={`h-4 w-4 ${tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
      </Button>
    </div>
  );
};

export default StockHeader;
