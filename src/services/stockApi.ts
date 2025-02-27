
import { Stock, NewsItem } from '@/utils/types';
import { mockStocks } from '@/utils/mockData';

// API URL'leri (gerçek kullanımda burada gerçek API endpointleri olacaktır)
const BIST_API_BASE_URL = 'https://api.example.com/bist';

// Gerçek API'niz yoksa, aşağıdaki mock gecikme simulasyonu ile verileri döndürebilirsiniz
const mockDelay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export interface StockApiResponse {
  success: boolean;
  data?: Stock[];
  error?: string;
}

export interface StockDetailResponse {
  success: boolean;
  data?: Stock;
  error?: string;
}

export const stockApi = {
  /**
   * Tüm BIST hisselerini getirir
   */
  getAllStocks: async (): Promise<StockApiResponse> => {
    try {
      // Gerçek API'niz olduğunda, aşağıdaki gibi bir fetch isteği yapabilirsiniz:
      // const response = await fetch(`${BIST_API_BASE_URL}/stocks`);
      // const data = await response.json();
      // return { success: true, data };
      
      // Mock veri için:
      await mockDelay(800); // API çağrısını simüle etmek için gecikme
      
      // Stok verilerini döndür
      return {
        success: true,
        data: mockStocks.map(stock => ({
          ...stock,
          // Fiyatı küçük bir rasgele değerle güncelle (gerçek zamanlı simulasyonu için)
          price: stock.price * (1 + (Math.random() * 0.06 - 0.03)), // ±%3 fiyat dalgalanması
          priceChange: stock.priceChange * (1 + (Math.random() * 0.1 - 0.05)) // ±%5 değişim dalgalanması
        }))
      };
    } catch (error) {
      console.error('Hisse verileri alınırken hata oluştu:', error);
      return {
        success: false,
        error: 'Hisse verileri alınamadı. Lütfen daha sonra tekrar deneyin.'
      };
    }
  },

  /**
   * Belirli bir hissenin detaylarını getirir
   */
  getStockDetails: async (symbol: string): Promise<StockDetailResponse> => {
    try {
      // Gerçek API'niz olduğunda:
      // const response = await fetch(`${BIST_API_BASE_URL}/stocks/${symbol}`);
      // const data = await response.json();
      // return { success: true, data };
      
      // Mock veri için:
      await mockDelay(600);
      
      const stockData = mockStocks.find(stock => stock.symbol === symbol);
      
      if (!stockData) {
        return {
          success: false,
          error: `${symbol} sembolü ile hisse bulunamadı.`
        };
      }
      
      // Fiyatı küçük bir rasgele değerle güncelle (gerçek zamanlı simulasyonu için)
      const updatedStock = {
        ...stockData,
        price: stockData.price * (1 + (Math.random() * 0.04 - 0.02)), // ±%2 fiyat dalgalanması
        priceChange: stockData.priceChange * (1 + (Math.random() * 0.08 - 0.04)) // ±%4 değişim dalgalanması
      };
      
      return {
        success: true,
        data: updatedStock
      };
    } catch (error) {
      console.error(`${symbol} hisse detayları alınırken hata oluştu:`, error);
      return {
        success: false,
        error: 'Hisse detayları alınamadı. Lütfen daha sonra tekrar deneyin.'
      };
    }
  },

  /**
   * Belirli bir hissenin haberlerini getirir
   */
  getStockNews: async (symbol: string): Promise<NewsItem[]> => {
    try {
      // Gerçek API'niz olduğunda:
      // const response = await fetch(`${BIST_API_BASE_URL}/stocks/${symbol}/news`);
      // const data = await response.json();
      // return data;
      
      // Mock veri için:
      await mockDelay(700);
      
      const stockData = mockStocks.find(stock => stock.symbol === symbol);
      
      if (!stockData) {
        return [];
      }
      
      return stockData.news;
    } catch (error) {
      console.error(`${symbol} hisse haberleri alınırken hata oluştu:`, error);
      return [];
    }
  }
};
