
import React, { useEffect, useState } from 'react';
import { Stock, NewsItem } from '@/utils/types';
import StockCard from '@/components/StockCard';
import EmptyState from '@/components/EmptyState';
import { checkForNewsAndNotifyUser } from '@/services/newsService';
import { toast } from 'sonner';

type TrackedStocksListProps = {
  stocks: Stock[];
  loading: boolean;
  onToggleTracking: (id: string) => void;
  onSearchClick: () => void;
};

const TrackedStocksList = ({ 
  stocks, 
  loading, 
  onToggleTracking, 
  onSearchClick 
}: TrackedStocksListProps) => {
  const [lastCheckTimestamp, setLastCheckTimestamp] = useState<string | null>(null);
  const trackedStocks = stocks.filter(stock => stock.tracked);
  const hasNewsInTrackedStocks = trackedStocks.some(stock => stock.news.length > 0);

  useEffect(() => {
    // Check for new news on component mount
    checkForLatestNews();
    
    // Set up interval to check for news every 5 minutes
    const intervalId = setInterval(checkForLatestNews, 5 * 60 * 1000);
    
    return () => clearInterval(intervalId);
  }, []);
  
  const checkForLatestNews = async () => {
    try {
      const response = await fetch('/news_archive.json');
      const data = await response.json();
      
      // If this is the first check or if the timestamp has changed, update news
      if (!lastCheckTimestamp || lastCheckTimestamp !== data.timestamp) {
        console.log('News file updated, checking for new stock news');
        setLastCheckTimestamp(data.timestamp);
        
        // This will check if there are news for tracked stocks and show notifications
        const hasNews = await checkForNewsAndNotifyUser();
        
        if (!hasNews) {
          console.log('No new news for tracked stocks found');
        }
      }
    } catch (error) {
      console.error('Error checking latest news:', error);
    }
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="border rounded-lg h-64 animate-pulse bg-muted"></div>
        ))}
      </div>
    );
  }

  if (trackedStocks.length === 0) {
    return <EmptyState onSearchClick={onSearchClick} />;
  }

  return (
    <>
      {!hasNewsInTrackedStocks && (
        <div className="p-8 text-center bg-muted rounded-lg mb-6">
          <p className="text-lg font-medium">Takip ettiğiniz hisselerle ilgili yeni bir gelişme yok.</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {trackedStocks.map(stock => (
          <StockCard
            key={stock.id}
            stock={stock}
            onToggleTracking={onToggleTracking}
          />
        ))}
      </div>
    </>
  );
};

export default TrackedStocksList;
