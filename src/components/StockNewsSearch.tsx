
import React, { useState } from 'react';
import { Search, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { GoogleSearchResult } from '@/utils/types';
import { toast } from 'sonner';

interface StockNewsSearchProps {
  stockSymbol: string;
  stockName: string;
  onResultsFound: (results: GoogleSearchResult[]) => void;
}

const StockNewsSearch: React.FC<StockNewsSearchProps> = ({ 
  stockSymbol, 
  stockName, 
  onResultsFound 
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const searchNews = async () => {
    setIsLoading(true);
    try {
      // For demo purposes, we'll create mock results
      // In a real implementation, you would make an API call to Google Custom Search API
      // Example API URL: https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_SEARCH_ENGINE_ID&q=SEARCH_QUERY
      
      // Mock search results for demonstration
      const mockResults: GoogleSearchResult[] = [
        {
          title: `${stockName} (${stockSymbol}) Reports Strong Q2 Earnings`,
          link: "https://example.com/news/1",
          snippet: `${stockName} announced quarterly earnings above analyst expectations, driving the stock price up by 5%.`,
          source: "Financial Times",
          publishedTime: new Date(Date.now() - 86400000).toISOString() // 1 day ago
        },
        {
          title: `Analysts Upgrade ${stockSymbol} Stock Rating`,
          link: "https://example.com/news/2",
          snippet: `Several analysts have upgraded their outlook on ${stockName}, citing positive growth prospects and strong market position.`,
          source: "Bloomberg",
          publishedTime: new Date(Date.now() - 172800000).toISOString() // 2 days ago
        },
        {
          title: `${stockName} Expands Operations with New Facility`,
          link: "https://example.com/news/3",
          snippet: `${stockName} announced the opening of a new production facility, expected to increase capacity by 30% by end of year.`,
          source: "Reuters",
          publishedTime: new Date(Date.now() - 259200000).toISOString() // 3 days ago
        }
      ];

      // In a real implementation, you would parse the Google API response
      // const response = await fetch(`https://www.googleapis.com/customsearch/v1?key=${apiKey}&cx=${searchEngineId}&q=${stockSymbol}+${stockName}+stock+news`);
      // const data = await response.json();
      // const results = data.items.map(item => ({
      //   title: item.title,
      //   link: item.link,
      //   snippet: item.snippet,
      //   source: item.displayLink,
      //   publishedTime: item.pagemap?.newsarticle?.[0]?.datepublished || new Date().toISOString()
      // }));

      onResultsFound(mockResults);
      toast.success(`Found ${mockResults.length} news articles for ${stockSymbol}`);
    } catch (error) {
      console.error('Error searching for news:', error);
      toast.error('Failed to fetch news articles');
    } finally {
      setIsLoading(false);
      setIsOpen(false);
    }
  };

  return (
    <div className="mb-4">
      <Button 
        variant="outline" 
        size="sm" 
        className="w-full flex justify-between items-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>Search Latest News</span>
        <Search className="h-4 w-4" />
      </Button>
      
      {isOpen && (
        <div className="mt-2 p-4 border rounded-md shadow-sm bg-card">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-medium">Search News for {stockSymbol}</h4>
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-6 w-6"
              onClick={() => setIsOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <p className="text-sm text-muted-foreground mb-4">
            Find the latest news articles about {stockName} ({stockSymbol})
          </p>
          
          <Button
            onClick={searchNews}
            disabled={isLoading}
            className="w-full"
          >
            {isLoading ? 'Searching...' : 'Search Google News'}
          </Button>
        </div>
      )}
    </div>
  );
};

export default StockNewsSearch;
