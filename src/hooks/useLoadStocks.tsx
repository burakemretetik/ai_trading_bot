
import { useState, useEffect } from 'react';
import { Stock } from '@/utils/types';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { getTrackedStocks } from '@/services/stockService';
import { checkForNewsAndNotifyUser } from '@/services/newsService';
import { toast } from 'sonner';

export function useLoadStocks() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [newsArchiveTimestamp, setNewsArchiveTimestamp] = useState<string | null>(null);
  const [isFollowAllActive, setIsFollowAllActive] = useState(false);

  const loadStocks = async () => {
    setLoading(true);
    try {
      // Try to load stocks from CSV
      const csvStocks = await createMockStocksFromCSV();
      
      // Get tracked stocks from localStorage
      const trackedStockIds = await getTrackedStocks();
      
      console.log('Tracked stock IDs:', trackedStockIds);
      
      // Use CSV data if available, otherwise fall back to mock data
      if (csvStocks && csvStocks.length > 0) {
        // Update tracking status based on localStorage data
        const updatedStocks = csvStocks.map(stock => ({
          ...stock,
          tracked: trackedStockIds.includes(stock.id)
        }));
        
        setStocks(updatedStocks);
        
        // Check if all stocks are being tracked
        const allTracked = updatedStocks.length > 0 && 
          updatedStocks.every(stock => stock.tracked);
        setIsFollowAllActive(allTracked);
      } else {
        // For mock data, also check localStorage
        const updatedMockStocks = mockStocks.map(stock => ({
          ...stock,
          tracked: trackedStockIds.includes(stock.id)
        }));
        
        setStocks(updatedMockStocks);
        
        // Check if all stocks are being tracked in mock data
        const allTracked = updatedMockStocks.length > 0 && 
          updatedMockStocks.every(stock => stock.tracked);
        setIsFollowAllActive(allTracked);
      }
      
      // Check for news updates and notify user if there are any
      const hasNews = await checkForNewsAndNotifyUser();
      if (hasNews) {
        console.log('WhatsApp notifications sent to user for news updates');
      }
      
      // Get news archive timestamp
      try {
        const response = await fetch('/news_archive.json');
        const data = await response.json();
        setNewsArchiveTimestamp(data.timestamp);
      } catch (error) {
        console.error('Error fetching news archive timestamp:', error);
      }
    } catch (error) {
      console.error('Error loading stocks:', error);
      
      // For error fallback, also check localStorage
      const trackedStockIds = await getTrackedStocks();
      
      const updatedMockStocks = mockStocks.map(stock => ({
        ...stock,
        tracked: trackedStockIds.includes(stock.id)
      }));
      
      setStocks(updatedMockStocks);
      
      // Check if all stocks are being tracked in case of error
      const allTracked = updatedMockStocks.length > 0 && 
        updatedMockStocks.every(stock => stock.tracked);
      setIsFollowAllActive(allTracked);
      
      toast.error('Hisse verileri yüklenirken bir hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  return {
    stocks,
    setStocks,
    loading,
    loadStocks,
    newsArchiveTimestamp,
    setNewsArchiveTimestamp,
    isFollowAllActive,
    setIsFollowAllActive
  };
}
