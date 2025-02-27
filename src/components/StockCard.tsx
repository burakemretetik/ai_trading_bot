
import React, { useState } from 'react';
import { Star } from 'lucide-react';
import { Stock, GoogleSearchResult, NewsItem } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItem from './NewsItem';
import StockNewsSearch from './StockNewsSearch';
import { formatDistanceToNow } from 'date-fns';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockCard: React.FC<StockCardProps> = ({ stock, onToggleTracking }) => {
  const [searchResults, setSearchResults] = useState<GoogleSearchResult[]>([]);
  
  const handleSearchResults = (results: GoogleSearchResult[]) => {
    setSearchResults(results);
  };
  
  // Convert Google search results to NewsItem format
  const searchResultsToNewsItems = (results: GoogleSearchResult[]): NewsItem[] => {
    return results.map((result, index) => ({
      id: `search-${index}-${Date.now()}`,
      title: result.title,
      source: result.source,
      url: result.link,
      publishedAt: result.publishedTime || new Date().toISOString(),
      summary: result.snippet
    }));
  };
  
  // Combined news: search results + stock news
  const combinedNews = [
    ...searchResultsToNewsItems(searchResults),
    ...stock.news
  ];

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
      
      <div className="p-4">
        <StockNewsSearch
          stockSymbol={stock.symbol}
          stockName={stock.name}
          onResultsFound={handleSearchResults}
        />
        
        <h4 className="text-sm font-medium mb-3">Latest News</h4>
        {combinedNews.length > 0 ? (
          combinedNews.map(item => (
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
