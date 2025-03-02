
import { StockNewsMapping } from './types';

export async function getStockNewsMapping(): Promise<StockNewsMapping | null> {
  try {
    const response = await fetch('/stock_news_mapping.json');
    if (!response.ok) {
      console.error('Failed to fetch stock news mapping:', response.statusText);
      return null;
    }
    
    const data: StockNewsMapping = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching stock news mapping:', error);
    return null;
  }
}

export function getNewsUrlsForStock(mapping: StockNewsMapping, stockName: string): string[] {
  if (!mapping.updated || !mapping.stock_news[stockName]) {
    return [];
  }
  
  return mapping.stock_news[stockName];
}
