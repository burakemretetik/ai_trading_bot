
import { StockNewsMapping, Stock } from '@/utils/types';
import { getStockNewsMapping } from '@/utils/stockNewsMapping';
import { toast } from 'sonner';

export async function checkForNewsAndNotifyUser(userStocks: Stock[]): Promise<boolean> {
  try {
    // Fetch the current mapping
    const mapping = await getStockNewsMapping();
    
    if (!mapping || !mapping.updated) {
      console.log('No news updates available');
      return false;
    }
    
    // Check if any of the user's tracked stocks have news
    const relevantNews: Record<string, string[]> = {};
    let hasRelevantNews = false;
    
    userStocks.forEach(stock => {
      if (stock.tracked && mapping.stock_news[stock.name]) {
        relevantNews[stock.name] = mapping.stock_news[stock.name];
        hasRelevantNews = true;
      }
    });
    
    if (hasRelevantNews) {
      // In a real app, this would trigger an email through a backend service
      console.log('Relevant news found for user stocks:', relevantNews);
      toast.info('Takip ettiğiniz hisseler hakkında yeni haberler var!');
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('Error checking for news updates:', error);
    return false;
  }
}

export async function formatNewsForEmail(userStocks: Stock[]): Promise<string> {
  const mapping = await getStockNewsMapping();
  if (!mapping || !mapping.updated) {
    return '';
  }
  
  let emailContent = '<h2>Takip Ettiğiniz Hisseler Hakkında Güncel Haberler</h2>';
  let hasContent = false;
  
  userStocks.forEach(stock => {
    if (stock.tracked && mapping.stock_news[stock.name]) {
      hasContent = true;
      emailContent += `<h3>${stock.name} (${stock.symbol})</h3><ul>`;
      
      mapping.stock_news[stock.name].forEach(url => {
        emailContent += `<li><a href="${url}">${url}</a></li>`;
      });
      
      emailContent += '</ul>';
    }
  });
  
  return hasContent ? emailContent : '';
}
