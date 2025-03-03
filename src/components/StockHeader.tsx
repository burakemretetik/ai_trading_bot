
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
      <div className="flex items-center space-x-3">
        <div className="flex flex-col">
          <div className="flex items-center">
            <h3 className="font-semibold text-lg">{symbol}</h3>
          </div>
          <p className="text-sm text-muted-foreground">{name}</p>
        </div>
      </div>
      
      <Button variant="ghost" size="icon" className="rounded-full" onClick={() => onToggleTracking(stockId)}>
        <Star className={`h-5 w-5 ${tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
      </Button>
    </div>
  );
};

export default StockHeader;
