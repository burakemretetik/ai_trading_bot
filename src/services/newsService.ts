
import { StockNewsMapping } from '@/utils/types';
import stockNewsMapping from '@/utils/stock_news_mapping.json';
import { toast } from 'sonner';
import { getTrackedStocks } from './stockService';

// This function will check if there's any news for stocks the user is tracking
export async function checkForNewsAndNotifyUser(): Promise<boolean> {
  try {
    const mapping = stockNewsMapping as StockNewsMapping;
    
    // Check if there are any news updates
    if (!mapping.updated || Object.keys(mapping.stock_news).length === 0) {
      console.log('No news updates available');
      return false;
    }
    
    // Get user's tracked stocks
    const trackedStockIds = await getTrackedStocks();
    
    // This would need to be adapted to your actual stock data structure
    // For now, we'll assume stock IDs and symbols are the same for simplicity
    const trackedSymbols = trackedStockIds;
    
    // Find news for tracked stocks
    const relevantNews: Record<string, string[]> = {};
    for (const symbol of trackedSymbols) {
      if (mapping.stock_news[symbol] && mapping.stock_news[symbol].length > 0) {
        relevantNews[symbol] = mapping.stock_news[symbol];
      }
    }
    
    // If there's relevant news, notify the user
    if (Object.keys(relevantNews).length > 0) {
      const stocksWithNews = Object.keys(relevantNews).join(', ');
      const totalNews = Object.values(relevantNews).flat().length;
      
      toast.info(
        `${totalNews} yeni haber bulundu: ${stocksWithNews}`, 
        { duration: 5000 }
      );
      
      // In a real implementation, this is where you'd call the email service
      // For now, we'll just return true to indicate there was news
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('Error in checkForNewsAndNotifyUser:', error);
    return false;
  }
}

// Helper to get news URLs for a specific stock
export function getNewsUrlsForStock(symbol: string): string[] {
  const mapping = stockNewsMapping as StockNewsMapping;
  return mapping.stock_news[symbol] || [];
}

// Helper to format news for email
export function formatNewsForEmail(stockNews: Record<string, string[]>): string {
  let emailContent = `<h1>Takip Ettiğiniz Hisseler İçin Son Haberler</h1>`;
  
  for (const [symbol, urls] of Object.entries(stockNews)) {
    emailContent += `<h2>${symbol}</h2><ul>`;
    for (const url of urls) {
      emailContent += `<li><a href="${url}">${url}</a></li>`;
    }
    emailContent += `</ul>`;
  }
  
  return emailContent;
}
