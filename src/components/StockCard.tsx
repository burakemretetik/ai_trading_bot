import React, { useState, useEffect } from 'react';
import { Star, RefreshCw, Globe } from 'lucide-react';
import { Stock, NewsItem } from '@/utils/types';
import { Button } from '@/components/ui/button';
import NewsItemComponent from './NewsItem';
import { supabase } from "@/integrations/supabase/client";
import { toast } from 'sonner';
import { getNewsUrlsForStock } from '@/services/newsService';
import { Card, CardContent, CardHeader, CardFooter } from '@/components/ui/card';

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
    fetchNewsFromDB();

    fetchNewsUrls();

    const checkForUpdates = async () => {
      try {
        const response = await fetch('/news_archive.json');
        const data = await response.json();

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
    const intervalId = setInterval(checkForUpdates, 60 * 1000);

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
        const newsItems: NewsItem[] = data.map(item => ({
          id: item.id,
          title: item.title,
          source: item.source,
          url: item.url,
          publishedAt: item.published_at,
          summary: item.summary,
          sentiment: (Math.random() > 0.6 ? 'positive' : Math.random() > 0.3 ? 'negative' : 'neutral') as "positive" | "negative" | "neutral",
          signalStrength: (Math.random() > 0.7 ? 'strong' : Math.random() > 0.4 ? 'medium' : Math.random() > 0.2 ? 'weak' : 'neutral') as "strong" | "medium" | "weak" | "neutral"
        }));
        setNews(newsItems);
      } else {
        const newsUrls = await getNewsUrlsForStock(stock.symbol);
        if (newsUrls && newsUrls.length > 0) {
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

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex flex-col">
              <div className="flex items-center">
                <h3 className="font-semibold text-lg">{stock.symbol}</h3>
              </div>
              <p className="text-sm text-muted-foreground">{stock.name}</p>
            </div>
          </div>
          
          <Button variant="ghost" size="icon" className="rounded-full" onClick={() => onToggleTracking(stock.id)}>
            <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="pt-2 flex-grow">
        {news.length > 0 ? (
          <div className="space-y-3 mt-2">
            {news.slice(0, 3).map(item => (
              <NewsItemComponent key={item.id} news={item} />
            ))}
          </div>
        ) : newsUrls.length > 0 ? (
          <div className="space-y-2">
            {newsUrls.slice(0, 3).map((url, index) => (
              <a 
                key={index} 
                href={url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="block p-2 border rounded hover:bg-muted transition-colors flex items-center"
              >
                <Globe className="h-4 w-4 mr-2" />
                <span className="text-sm truncate">{new URL(url).hostname.replace('www.', '')}</span>
              </a>
            ))}
          </div>
        ) : (
          <div className="flex-grow"></div>
        )}
      </CardContent>
      
      {(news.length > 0 || newsUrls.length > 0) && (
        <CardFooter className="pt-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleRefreshNews} 
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <RefreshCw className="h-4 w-4 mr-2" />
                <span>Refresh News</span>
              </>
            )}
          </Button>
        </CardFooter>
      )}
    </Card>
  );
};

export default StockCard;
