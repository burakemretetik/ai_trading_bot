
import { useState, useEffect } from 'react';
import { Stock } from '@/utils/types';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { getTrackedStocks, trackStock, untrackStock } from '@/services/stockService';
import { checkForNewsAndNotifyUser } from '@/services/newsService';
import { toast } from 'sonner';

export function useStocks() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [newsArchiveTimestamp, setNewsArchiveTimestamp] = useState<string | null>(null);
  
  useEffect(() => {
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
        } else {
          // For mock data, also check localStorage
          const updatedMockStocks = mockStocks.map(stock => ({
            ...stock,
            tracked: trackedStockIds.includes(stock.id)
          }));
          
          setStocks(updatedMockStocks);
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
        toast.error('Hisse verileri yüklenirken bir hata oluştu');
      } finally {
        setLoading(false);
      }
    };
    
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
    const stockToUpdate = stocks.find(stock => stock.id === id);
    if (!stockToUpdate) return;
    
    setStocks(prevStocks => 
      prevStocks.map(stock => {
        if (stock.id === id) {
          const newTrackedState = !stock.tracked;
          return { ...stock, tracked: newTrackedState };
        }
        return stock;
      })
    );
    
    try {
      if (!stockToUpdate.tracked) {
        // Track the stock
        const success = await trackStock(id);
        if (success) {
          toast.success(`${stockToUpdate.symbol} hisse takibinize eklendi`);
        }
      } else {
        // Untrack the stock
        const success = await untrackStock(id);
        if (success) {
          toast(`${stockToUpdate.symbol} hisse takibinizden çıkarıldı`);
        }
      }
    } catch (error) {
      console.error('Error toggling stock tracking:', error);
      
      // Revert the UI change if the API call failed
      setStocks(prevStocks => 
        prevStocks.map(stock => {
          if (stock.id === id) {
            return { ...stock, tracked: stockToUpdate.tracked };
          }
          return stock;
        })
      );
      
      toast.error('İşlem sırasında bir hata oluştu');
    }
  };
  
  const handleAddStock = async (stock: Stock) => {
    if (!stocks.some(s => s.id === stock.id)) {
      setStocks(prev => [...prev, stock]);
    }
    
    handleToggleTracking(stock.id);
  };

  return {
    stocks,
    loading,
    handleToggleTracking,
    handleAddStock
  };
}
