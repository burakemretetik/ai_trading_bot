
import { useEffect } from 'react';
import { Stock } from '@/utils/types';
import { getTrackedStocks } from '@/services/stockService';
import { useLoadStocks } from './useLoadStocks';
import { toggleStockTracking, toggleFollowAll } from '@/utils/stockTrackingUtils';

export function useStocks() {
  const { 
    stocks, 
    setStocks, 
    loading, 
    loadStocks, 
    newsArchiveTimestamp, 
    setNewsArchiveTimestamp,
    isFollowAllActive,
    setIsFollowAllActive
  } = useLoadStocks();
  
  useEffect(() => {
    loadStocks();
    
    // Set up an interval to check for news archive updates
    const checkNewsArchiveInterval = setInterval(async () => {
      try {
        const response = await fetch('/news_archive.json');
        const data = await response.json();
        
        if (newsArchiveTimestamp !== data.timestamp) {
          console.log('News archive updated, refreshing stocks');
          setNewsArchiveTimestamp(data.timestamp);
          loadStocks();
        }
      } catch (error) {
        console.error('Error checking news archive updates:', error);
      }
    }, 5 * 60 * 1000); // Check every 5 minutes
    
    return () => clearInterval(checkNewsArchiveInterval);
  }, []);

  const handleToggleTracking = async (id: string) => {
    await toggleStockTracking(id, stocks, setStocks, setIsFollowAllActive);
  };
  
  const handleAddStock = async (stock: Stock) => {
    if (!stocks.some(s => s.id === stock.id)) {
      setStocks(prev => [...prev, stock]);
    }
    
    handleToggleTracking(stock.id);
  };
  
  const handleToggleFollowAll = async () => {
    await toggleFollowAll(isFollowAllActive, setIsFollowAllActive, stocks, setStocks, getTrackedStocks);
  };

  return {
    stocks,
    loading,
    handleToggleTracking,
    handleAddStock,
    isFollowAllActive,
    handleToggleFollowAll
  };
}
