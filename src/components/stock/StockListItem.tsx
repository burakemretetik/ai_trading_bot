
import React from 'react';
import { Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Stock } from '@/utils/types';

interface StockListItemProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockListItem: React.FC<StockListItemProps> = ({ stock, onToggleTracking }) => {
  return (
    <div className="grid grid-cols-12 px-4 py-4 border-b last:border-b-0 items-center hover:bg-muted/30 transition-colors">
      <div className="col-span-3 font-medium">{stock.symbol}</div>
      <div className="col-span-7 text-sm truncate" title={stock.name}>
        {stock.name}
      </div>
      <div className="col-span-2 flex justify-center">
        <Button 
          variant="ghost" 
          size="icon" 
          className="rounded-full" 
          onClick={() => onToggleTracking(stock.id)}
        >
          <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
          <span className="sr-only">
            {stock.tracked ? 'Takipten çıkar' : 'Takip et'}
          </span>
        </Button>
      </div>
    </div>
  );
};

export default StockListItem;
