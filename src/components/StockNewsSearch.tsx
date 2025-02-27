
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';
import { NewsItem, Stock, GoogleSearchResult, SignalStrength } from '@/utils/types';
import { toast } from 'sonner';
import NewsItem from './NewsItem';

interface StockNewsSearchProps {
  stock: Stock;
  onAddNewsToStock: (stockId: string, newsItem: NewsItem) => void;
}

const StockNewsSearch: React.FC<StockNewsSearchProps> = ({ stock, onAddNewsToStock }) => {
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<NewsItem[]>([]);
  const [searchQuery, setSearchQuery] = useState(`${stock.symbol} ${stock.name} stock news`);

  const handleSearch = async () => {
    setIsSearching(true);
    
    try {
      // This is a mock function that simulates Google search results
      // In a real implementation, you would connect to a backend API or Google Custom Search API
      const results = await mockGoogleSearchForStockNews(searchQuery);
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching for news:', error);
      toast.error('Haber araması sırasında bir hata oluştu');
    } finally {
      setIsSearching(false);
    }
  };

  const handleAddNews = (newsItem: NewsItem) => {
    onAddNewsToStock(stock.id, newsItem);
    toast.success(`Haber "${stock.symbol}" hissesine eklendi`);
  };

  // This is a mock function that simulates Google search results
  // Replace this with actual API call in production
  const mockGoogleSearchForStockNews = async (query: string): Promise<NewsItem[]> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Mock results based on the stock symbol in the query
    const mockResults: GoogleSearchResult[] = [
      {
        title: `${stock.symbol} Reports Strong Quarterly Earnings`,
        link: 'https://example.com/news/1',
        snippet: `${stock.name} announced quarterly earnings that exceeded analyst expectations, driving the stock price up by 5%.`,
        source: 'Financial Times',
        publishedTime: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      },
      {
        title: `New Product Launch from ${stock.symbol}`,
        link: 'https://example.com/news/2',
        snippet: `${stock.name} unveiled a new product line that analysts expect will significantly impact its market position.`,
        source: 'Bloomberg',
        publishedTime: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString()
      },
      {
        title: `Market Analysis: Is ${stock.symbol} a Good Investment?`,
        link: 'https://example.com/news/3',
        snippet: `Industry experts weigh in on whether ${stock.name} represents a good investment opportunity in the current market.`,
        source: 'CNBC',
        publishedTime: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString()
      }
    ];
    
    // Convert Google search results to NewsItem format
    return mockResults.map((result, index) => {
      // Assign random signal strength for demo purposes
      const signals: SignalStrength[] = ['strong', 'medium', 'weak', 'neutral'];
      const randomSignal = signals[Math.floor(Math.random() * signals.length)];
      
      return {
        id: `search-${stock.id}-${index}`,
        title: result.title,
        source: result.source,
        url: result.link,
        publishedAt: result.publishedTime || new Date().toISOString(),
        summary: result.snippet,
        signalStrength: randomSignal
      };
    });
  };

  return (
    <div className="mt-4 p-4 border rounded-lg bg-card">
      <h4 className="text-sm font-medium mb-3">Search News for {stock.symbol}</h4>
      
      <div className="flex gap-2 mb-4">
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter search terms..."
          className="flex-1"
        />
        <Button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? 'Searching...' : <Search className="h-4 w-4" />}
        </Button>
      </div>
      
      {searchResults.length > 0 ? (
        <div className="space-y-4">
          <h5 className="text-sm font-medium">Search Results</h5>
          {searchResults.map(newsItem => (
            <div key={newsItem.id} className="relative">
              <NewsItem news={newsItem} />
              <Button 
                variant="outline" 
                size="sm" 
                className="absolute top-4 right-4"
                onClick={() => handleAddNews(newsItem)}
              >
                Add to Stock
              </Button>
            </div>
          ))}
        </div>
      ) : isSearching ? (
        <div className="py-4 text-center">
          <p className="text-muted-foreground">Searching for news...</p>
        </div>
      ) : (
        <div className="py-4 text-center">
          <p className="text-muted-foreground">Adjust your search terms and click the search button to find news</p>
        </div>
      )}
    </div>
  );
};

export default StockNewsSearch;
