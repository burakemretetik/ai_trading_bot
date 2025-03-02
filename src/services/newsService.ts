
import { StockNewsMapping } from '@/utils/types';
import stockNewsMapping from '@/utils/stock_news_mapping.json';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { getTrackedStocks } from './stockService';
import { getUserSettings, hasValidPhoneNumber, formatPhoneNumber } from '@/services/userSettingsService';

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
      
      // Get user settings to check if WhatsApp is enabled and has a valid phone number
      const userSettings = getUserSettings();
      
      // Only send WhatsApp notification if enabled and has a valid phone number
      if (userSettings.whatsappEnabled && hasValidPhoneNumber()) {
        await sendWhatsAppNotification(relevantNews);
      } else if (userSettings.whatsappEnabled && !hasValidPhoneNumber()) {
        // Remind user to set up their phone number if WhatsApp is enabled but no phone number
        toast.warning(
          'WhatsApp bildirimleri için telefon numaranızı ayarlar sayfasından eklemelisiniz',
          { duration: 7000 }
        );
      }
      
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
    // Get user phone number from settings
    const { phoneNumber } = getUserSettings();
    
    if (!phoneNumber) {
      console.log('No phone number found for WhatsApp notification');
      return;
    }
    
    // Format the WhatsApp content
    const whatsappContent = formatNewsForWhatsApp(stockNews);
    
    // Format phone number to international format
    const formattedPhoneNumber = formatPhoneNumber(phoneNumber);
    
    // Call the Supabase function to send the WhatsApp notification
    const { data, error } = await supabase.functions.invoke('send-stock-news-whatsapp', {
      body: {
        content: whatsappContent,
        stockNews,
        recipientPhoneNumber: formattedPhoneNumber
      }
    });
    
    if (error) {
      console.error('Error sending WhatsApp notification:', error);
      toast.error('WhatsApp bildirimi gönderilemedi');
    } else if (data && data.success) {
      console.log('WhatsApp notification sent successfully:', data);
      toast.success('WhatsApp bildirimi gönderildi');
    } else {
      console.error('WhatsApp notification failed:', data);
      toast.error('WhatsApp bildirimi gönderilemedi: ' + (data?.message || 'Bilinmeyen hata'));
    }
  } catch (error) {
    console.error('Error in sendWhatsAppNotification:', error);
    toast.error('WhatsApp bildirimi gönderilirken bir hata oluştu');
  }
}
