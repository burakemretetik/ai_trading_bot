
import React, { useState } from 'react';
import { Star, Search, ChevronDown, ChevronUp } from 'lucide-react';
import { Stock, NewsItem } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItemComponent from './NewsItem';
import StockNewsSearch from './StockNewsSearch';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
  onAddNewsToStock?: (stockId: string, newsItem: NewsItem) => void;
}

const StockCard: React.FC<StockCardProps> = ({ 
  stock, 
  onToggleTracking,
  onAddNewsToStock = () => {} // Default empty function if not provided
}) => {
  const [showSearch, setShowSearch] = useState(false);

  const toggleSearch = () => {
    setShowSearch(!showSearch);
  };

  return (
    <div className="border rounded-lg overflow-hidden bg-card animate-fade-in">
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex flex-col">
              <div className="flex items-center">
                <h3 className="font-semibold text-lg">{stock.symbol}</h3>
              </div>
              <p className="text-sm text-muted-foreground">{stock.name}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              className="rounded-full"
              onClick={toggleSearch}
              title="Search for news"
            >
              <Search className="h-4 w-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="icon"
              className="rounded-full"
              onClick={() => onToggleTracking(stock.id)}
            >
              <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
            </Button>
          </div>
        </div>
      </div>
      
      {showSearch && (
        <StockNewsSearch 
          stock={stock} 
          onAddNewsToStock={onAddNewsToStock} 
        />
      )}
      
      <div className="p-4">
        <h4 className="text-sm font-medium mb-3">Latest News</h4>
        {stock.news.length > 0 ? (
          stock.news.map(item => (
            <NewsItemComponent key={item.id} news={item} />
          ))
        ) : (
          <p className="text-sm text-muted-foreground py-4 text-center">No recent news available</p>
        )}
      </div>
    </div>
  );
};

export default StockCard;
