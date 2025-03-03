
import { Stock } from '@/utils/types';
import { trackStock, untrackStock } from '@/services/stockService';
import { toast } from 'sonner';

export async function toggleStockTracking(
  stockId: string, 
  stocks: Stock[], 
  setStocks: React.Dispatch<React.SetStateAction<Stock[]>>,
  setIsFollowAllActive: React.Dispatch<React.SetStateAction<boolean>>
) {
  const stockToUpdate = stocks.find(stock => stock.id === stockId);
  if (!stockToUpdate) return;
  
  setStocks(prevStocks => 
    prevStocks.map(stock => {
      if (stock.id === stockId) {
        const newTrackedState = !stock.tracked;
        return { ...stock, tracked: newTrackedState };
      }
      return stock;
    })
  );
  
  try {
    if (!stockToUpdate.tracked) {
      // Track the stock
      const success = await trackStock(stockId);
      if (success) {
        toast.success(`${stockToUpdate.symbol} hisse takibinize eklendi`);
      }
    } else {
      // Untrack the stock
      const success = await untrackStock(stockId);
      if (success) {
        toast(`${stockToUpdate.symbol} hisse takibinizden çıkarıldı`);
      }
    }
    
    // Update follow all toggle state
    const updatedStocks = stocks.map(stock => 
      stock.id === stockId ? { ...stock, tracked: !stockToUpdate.tracked } : stock
    );
    const allTracked = updatedStocks.length > 0 && 
      updatedStocks.every(stock => stock.tracked);
    setIsFollowAllActive(allTracked);
    
  } catch (error) {
    console.error('Error toggling stock tracking:', error);
    
    // Revert the UI change if the API call failed
    setStocks(prevStocks => 
      prevStocks.map(stock => {
        if (stock.id === stockId) {
          return { ...stock, tracked: stockToUpdate.tracked };
        }
        return stock;
      })
    );
    
    toast.error('İşlem sırasında bir hata oluştu');
  }
}

export async function toggleFollowAll(
  isFollowAllActive: boolean,
  setIsFollowAllActive: React.Dispatch<React.SetStateAction<boolean>>,
  stocks: Stock[],
  setStocks: React.Dispatch<React.SetStateAction<Stock[]>>,
  getTrackedStocks: () => Promise<string[]>
) {
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
}
