
import React, { useState, useEffect } from 'react';
import { Star, RefreshCw } from 'lucide-react';
import { Stock, NewsItem } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItemComponent from './NewsItem';
import { supabase } from "@/integrations/supabase/client";
import { toast } from 'sonner';
import { getNewsUrlsForStock } from '@/services/newsService';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockCard: React.FC<StockCardProps> = ({ stock, onToggleTracking }) => {
  const [loading, setLoading] = useState(false);
  const [news, setNews] = useState<NewsItem[]>(stock.news);
  const [lastNewsUpdate, setLastNewsUpdate] = useState<string | null>(null);
  
  useEffect(() => {
    // Get news from database
    fetchNewsFromDB();
    
    // Set up interval to check for news updates every minute
    const checkForUpdates = async () => {
      try {
        const response = await fetch('/news_archive.json');
        const data = await response.json();
        
        // If the timestamp has changed, refresh news
        if (lastNewsUpdate !== data.timestamp) {
          console.log(`News archive updated for ${stock.symbol}, refreshing news`);
          setLastNewsUpdate(data.timestamp);
          fetchNewsFromDB();
        }
      } catch (error) {
        console.error('Error checking for news updates:', error);
      }
    };
    
    const intervalId = setInterval(checkForUpdates, 60 * 1000); // Check every minute
    
    return () => clearInterval(intervalId);
  }, [stock.id, lastNewsUpdate]);
  
  const fetchNewsFromDB = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('stock_news')
        .select('*')
        .eq('stock_symbol', stock.symbol)
        .order('published_at', { ascending: false });
      
      if (error) {
        console.error('Error fetching news from DB:', error);
        return;
      }
      
      if (data && data.length > 0) {
        // Transform the data to match our NewsItem type
        const newsItems: NewsItem[] = data.map(item => ({
          id: item.id,
          title: item.title,
          source: item.source,
          url: item.url,
          publishedAt: item.published_at,
          summary: item.summary,
          // Add mock sentiment and signal strength with correct types
          sentiment: (Math.random() > 0.6 ? 'positive' : Math.random() > 0.3 ? 'negative' : 'neutral') as "positive" | "negative" | "neutral",
          signalStrength: (Math.random() > 0.7 ? 'strong' : Math.random() > 0.4 ? 'medium' : Math.random() > 0.2 ? 'weak' : 'neutral') as "strong" | "medium" | "weak" | "neutral",
        }));
        
        setNews(newsItems);
      } else {
        // Also check if there are any news URLs in the stock_news_mapping.json file
        const newsUrls = await getNewsUrlsForStock(stock.symbol);
        
        if (newsUrls && newsUrls.length > 0) {
          // Create simplified news items from URLs
          const mappedNewsItems: NewsItem[] = newsUrls.map((url, index) => ({
            id: `mapped-${stock.symbol}-${index}`,
            title: `${stock.name} ile ilgili yeni haber`,
            source: new URL(url).hostname.replace('www.', ''),
            url: url,
            publishedAt: new Date().toISOString(),
            summary: 'Bu habere tıklayarak detayları görebilirsiniz.',
            sentiment: 'neutral' as "positive" | "negative" | "neutral",
            signalStrength: 'medium' as "strong" | "medium" | "weak" | "neutral",
          }));
          
          setNews(mappedNewsItems);
        }
      }
    } catch (err) {
      console.error('Error in fetchNewsFromDB:', err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleRefreshNews = async () => {
    setLoading(true);
    try {
      await fetchNewsFromDB();
      toast.success(`News refreshed for ${stock.symbol}`);
    } catch (error) {
      console.error('Error refreshing news:', error);
      toast.error('Failed to refresh news');
    } finally {
      setLoading(false);
    }
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
          
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="icon"
              className="rounded-full"
              onClick={handleRefreshNews}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              <span className="sr-only">Refresh news</span>
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
      
      <div className="p-4">
        <h4 className="text-sm font-medium mb-3">Latest News</h4>
        {news.length > 0 ? (
          news.map(item => (
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
