
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
  const [isFollowAllActive, setIsFollowAllActive] = useState(false);
  
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
      
      // Update follow all toggle state
      const updatedStocks = stocks.map(stock => 
        stock.id === id ? { ...stock, tracked: !stockToUpdate.tracked } : stock
      );
      const allTracked = updatedStocks.length > 0 && 
        updatedStocks.every(stock => stock.tracked);
      setIsFollowAllActive(allTracked);
      
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
  
  const handleToggleFollowAll = async () => {
    const newFollowAllState = !isFollowAllActive;
    setIsFollowAllActive(newFollowAllState);
    
    // First update the UI for immediate feedback
    setStocks(prev => prev.map(stock => ({
      ...stock,
      tracked: newFollowAllState
    })));
    
    // Then process the changes in the background
    let successCount = 0;
    let failCount = 0;
    
    try {
      // Create an array of promises for all the tracking operations
      const operations = stocks.map(async (stock) => {
        try {
          if (newFollowAllState && !stock.tracked) {
            // Track the stock if it's not already tracked
            const success = await trackStock(stock.id);
            if (success) successCount++;
            else failCount++;
          } else if (!newFollowAllState && stock.tracked) {
            // Untrack the stock if it's currently tracked
            const success = await untrackStock(stock.id);
            if (success) successCount++;
            else failCount++;
          }
        } catch (error) {
          console.error(`Error processing stock ${stock.symbol}:`, error);
          failCount++;
        }
      });
      
      // Wait for all operations to complete
      await Promise.all(operations);
      
      // Show success message
      if (successCount > 0) {
        if (newFollowAllState) {
          toast.success(`${successCount} hisse takibinize eklendi`);
        } else {
          toast(`${successCount} hisse takibinizden çıkarıldı`);
        }
      }
      
      // Show error message if any operations failed
      if (failCount > 0) {
        toast.error(`${failCount} hisse işlemi başarısız oldu`);
      }
    } catch (error) {
      console.error('Error in bulk operation:', error);
      toast.error('İşlem sırasında bir hata oluştu');
      
      // If the overall operation fails, reload the current state from localStorage
      const trackedStockIds = await getTrackedStocks();
      setStocks(prev => prev.map(stock => ({
        ...stock,
        tracked: trackedStockIds.includes(stock.id)
      })));
      
      // Update follow all toggle based on reloaded state
      const reloadedAllTracked = stocks.length > 0 && 
        stocks.every(stock => trackedStockIds.includes(stock.id));
      setIsFollowAllActive(reloadedAllTracked);
    }
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
