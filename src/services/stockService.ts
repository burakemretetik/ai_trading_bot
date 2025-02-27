
import { toast } from 'sonner';

// Since we've removed authentication, we'll store tracked stocks in localStorage
const TRACKED_STOCKS_KEY = 'trackedStocks';

// Helper to get tracked stocks from localStorage
const getTrackedStocksFromStorage = (): string[] => {
  const storedValue = localStorage.getItem(TRACKED_STOCKS_KEY);
  return storedValue ? JSON.parse(storedValue) : [];
};

// Helper to save tracked stocks to localStorage
const saveTrackedStocksToStorage = (stockIds: string[]) => {
  localStorage.setItem(TRACKED_STOCKS_KEY, JSON.stringify(stockIds));
};

export async function getTrackedStocks(): Promise<string[]> {
  try {
    return getTrackedStocksFromStorage();
  } catch (error) {
    console.error('Error in getTrackedStocks:', error);
    return [];
  }
}

export async function trackStock(stockId: string): Promise<boolean> {
  try {
    const trackedStocks = getTrackedStocksFromStorage();
    
    // Don't add if already tracked
    if (trackedStocks.includes(stockId)) {
      return true;
    }
    
    // Add to tracked stocks
    trackedStocks.push(stockId);
    saveTrackedStocksToStorage(trackedStocks);
    
    return true;
  } catch (error) {
    console.error('Error in trackStock:', error);
    toast.error('Hisse takip edilemedi');
    return false;
  }
}

export async function untrackStock(stockId: string): Promise<boolean> {
  try {
    const trackedStocks = getTrackedStocksFromStorage();
    const updatedTrackedStocks = trackedStocks.filter(id => id !== stockId);
    
    saveTrackedStocksToStorage(updatedTrackedStocks);
    
    return true;
  } catch (error) {
    console.error('Error in untrackStock:', error);
    toast.error('Hisse takipten çıkarılamadı');
    return false;
  }
}
