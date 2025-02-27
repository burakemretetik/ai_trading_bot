
import React from 'react';
import { ChevronUp, ChevronDown, Star, BellRing } from 'lucide-react';
import { Stock } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItem from './NewsItem';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockCard: React.FC<StockCardProps> = ({ stock, onToggleTracking }) => {
  const priceIsPositive = stock.priceChange >= 0;
  const hasStrongSignal = stock.news.some(n => n.signalStrength === 'strong');
  
  return (
    <div className="border rounded-lg overflow-hidden bg-card animate-fade-in">
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex flex-col">
              <div className="flex items-center">
                <h3 className="font-semibold text-lg">{stock.symbol}</h3>
                {hasStrongSignal && (
                  <div className="ml-2 bg-signal-strong/20 rounded-full p-1">
                    <BellRing className="h-3 w-3 text-signal-strong" />
                  </div>
                )}
              </div>
              <p className="text-sm text-muted-foreground">{stock.name}</p>
            </div>
          </div>
          
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full"
            onClick={() => onToggleTracking(stock.id)}
          >
            <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
          </Button>
        </div>
        
        <div className="mt-3 flex items-center">
          <span className="text-2xl font-semibold">${stock.price.toFixed(2)}</span>
          <div className={`flex items-center ml-2 ${priceIsPositive ? 'text-signal-strong' : 'text-signal-weak'}`}>
            {priceIsPositive ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            <span className="text-sm font-medium">{Math.abs(stock.priceChange).toFixed(2)}</span>
          </div>
        </div>
      </div>
      
      <div className="p-4">
        <h4 className="text-sm font-medium mb-3">Latest News & Signals</h4>
        {stock.news.length > 0 ? (
          stock.news.map(item => (
            <NewsItem key={item.id} news={item} />
          ))
        ) : (
          <p className="text-sm text-muted-foreground py-4 text-center">No recent news available</p>
        )}
      </div>
    </div>
  );
};

export default StockCard;
