
import { StockNewsMapping } from '@/utils/types';
import stockNewsMapping from '@/utils/stock_news_mapping.json';
import { supabase } from '@/integrations/supabase/client';
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
      
      // Send a WhatsApp notification with the news if there are relevant updates
      await sendWhatsAppNotification(relevantNews);
      
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

// Helper to format news for WhatsApp
function formatNewsForWhatsApp(stockNews: Record<string, string[]>): string {
  let whatsappContent = `*Takip Ettiğiniz Hisseler İçin Son Haberler*\n\n`;
  
  for (const [symbol, urls] of Object.entries(stockNews)) {
    whatsappContent += `*${symbol}*\n`;
    for (const url of urls) {
      whatsappContent += `- ${url}\n`;
    }
    whatsappContent += `\n`;
  }
  
  return whatsappContent;
}

// Send WhatsApp notification with stock news to the user
async function sendWhatsAppNotification(stockNews: Record<string, string[]>): Promise<void> {
  try {
    // Format the WhatsApp content
    const whatsappContent = formatNewsForWhatsApp(stockNews);
    
    // Call the Supabase function to send the WhatsApp notification
    const { error } = await supabase.functions.invoke('send-stock-news-whatsapp', {
      body: {
        content: whatsappContent,
        stockNews
      }
    });
    
    if (error) {
      console.error('Error sending WhatsApp notification:', error);
      toast.error('WhatsApp bildirimi gönderilemedi');
    } else {
      console.log('WhatsApp notification sent successfully');
      toast.success('WhatsApp bildirimi gönderildi');
    }
  } catch (error) {
    console.error('Error in sendWhatsAppNotification:', error);
    toast.error('WhatsApp bildirimi gönderilirken bir hata oluştu');
  }
}
