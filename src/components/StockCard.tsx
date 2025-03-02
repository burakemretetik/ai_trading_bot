import React, { useState, useEffect } from 'react';
import { Star, RefreshCw, Globe } from 'lucide-react';
import { Stock, NewsItem } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItemComponent from './NewsItem';
import { supabase } from "@/integrations/supabase/client";
import { toast } from 'sonner';
import { getNewsUrlsForStock } from '@/services/newsService';
import { Card, CardContent } from '@/components/ui/card';
interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}
const StockCard: React.FC<StockCardProps> = ({
  stock,
  onToggleTracking
}) => {
  const [loading, setLoading] = useState(false);
  const [news, setNews] = useState<NewsItem[]>(stock.news);
  const [lastNewsUpdate, setLastNewsUpdate] = useState<string | null>(null);
  const [newsUrls, setNewsUrls] = useState<string[]>([]);
  useEffect(() => {
    // Get news from database
    fetchNewsFromDB();

    // Fetch news URLs from stock_news_mapping.json
    fetchNewsUrls();

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
          fetchNewsUrls();
        }
      } catch (error) {
        console.error('Error checking for news updates:', error);
      }
    };
    const intervalId = setInterval(checkForUpdates, 60 * 1000); // Check every minute

    return () => clearInterval(intervalId);
  }, [stock.id, lastNewsUpdate]);
  const fetchNewsUrls = async () => {
    try {
      const urls = await getNewsUrlsForStock(stock.symbol);
      if (urls && urls.length > 0) {
        setNewsUrls(urls);
      }
    } catch (error) {
      console.error('Error fetching news URLs:', error);
    }
  };
  const fetchNewsFromDB = async () => {
    try {
      setLoading(true);
      const {
        data,
        error
      } = await supabase.from('stock_news').select('*').eq('stock_symbol', stock.symbol).order('published_at', {
        ascending: false
      });
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
          signalStrength: (Math.random() > 0.7 ? 'strong' : Math.random() > 0.4 ? 'medium' : Math.random() > 0.2 ? 'weak' : 'neutral') as "strong" | "medium" | "weak" | "neutral"
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
            signalStrength: 'medium' as "strong" | "medium" | "weak" | "neutral"
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
      await fetchNewsUrls();
      toast.success(`News refreshed for ${stock.symbol}`);
    } catch (error) {
      console.error('Error refreshing news:', error);
      toast.error('Failed to refresh news');
    } finally {
      setLoading(false);
    }
  };
  return <div className="border rounded-lg overflow-hidden bg-card animate-fade-in">
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
            <Button variant="ghost" size="icon" className="rounded-full" onClick={handleRefreshNews} disabled={loading}>
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              <span className="sr-only">Refresh news</span>
            </Button>
            
            <Button variant="ghost" size="icon" className="rounded-full" onClick={() => onToggleTracking(stock.id)}>
              <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
            </Button>
          </div>
        </div>
      </div>
      
      <div className="p-4">
        
        {news.length > 0 ? news.map(item => <NewsItemComponent key={item.id} news={item} />) : <p className="text-sm text-muted-foreground py-4 text-center">Yeni Bir Gelişme Yok</p>}

        {/* News URLs from news_archive.json */}
        {newsUrls.length > 0 && <div className="mt-4">
            
            <Card>
              <CardContent className="p-3">
                <ul className="space-y-2">
                  {newsUrls.map((url, index) => <li key={`archive-${index}`} className="text-sm flex items-center">
                      <Globe className="h-3 w-3 mr-2 text-muted-foreground" />
                      <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline truncate">
                        {new URL(url).hostname.replace('www.', '')}
                      </a>
                    </li>)}
                </ul>
              </CardContent>
            </Card>
          </div>}
      </div>
    </div>;
};
export default StockCard;