
import React, { useState, useEffect } from 'react';
import { Star, RefreshCw } from 'lucide-react';
import { Stock } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItem from './NewsItem';
import { supabase } from "@/integrations/supabase/client";
import { toast } from 'sonner';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockCard: React.FC<StockCardProps> = ({ stock, onToggleTracking }) => {
  const [loading, setLoading] = useState(false);
  const [news, setNews] = useState(stock.news);
  
  useEffect(() => {
    // Get news from database
    fetchNewsFromDB();
  }, [stock.id]);
  
  const fetchNewsFromDB = async () => {
    try {
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
        const newsItems = data.map(item => ({
          id: item.id,
          title: item.title,
          source: item.source,
          url: item.url,
          publishedAt: item.published_at,
          summary: item.summary,
          // Add mock sentiment and signal strength
          sentiment: Math.random() > 0.6 ? 'positive' : Math.random() > 0.3 ? 'negative' : 'neutral',
          signalStrength: Math.random() > 0.7 ? 'strong' : Math.random() > 0.4 ? 'medium' : Math.random() > 0.2 ? 'weak' : 'neutral',
        }));
        
        setNews(newsItems);
      }
    } catch (err) {
      console.error('Error in fetchNewsFromDB:', err);
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
