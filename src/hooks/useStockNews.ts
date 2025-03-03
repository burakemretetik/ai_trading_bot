
import { useState, useEffect } from 'react';
import { Stock, NewsItem } from '@/utils/types';
import { supabase } from "@/integrations/supabase/client";
import { toast } from 'sonner';
import { getNewsUrlsForStock } from '@/services/newsService';

export function useStockNews(stock: Stock) {
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

  return {
    news,
    newsUrls,
    loading,
    handleRefreshNews
  };
}
