
import { Stock, EmailSettings } from './types';

// Base mock news data that we'll reuse
const mockNewsItems = [];

// Function to create mock stocks from CSV data
export const createMockStocksFromCSV = async () => {
  try {
    // Dynamically import the parseCSV function to avoid circular dependencies
    const { parseCSV } = await import('./csvParser');
    // Update the file path to the correct location
    const csvData = await parseCSV('/bist_100_hisseleri.csv');
    
    // If no CSV data is found, immediately return empty array to use fallback
    if (!csvData || csvData.length === 0) {
      console.log('No CSV data found, using fallback mock data');
      return [];
    }
    
    console.log(`Loading all ${csvData.length} stocks from BIST 100 CSV`);
    
    // Create mock stocks from ALL CSV data (removed the slice)
    const stocks: Stock[] = csvData.map((item, index) => {
      // Assign a random tracked status (with some being tracked by default)
      const tracked = index < 5 || Math.random() > 0.8;
      
      return {
        id: (index + 1).toString(),
        symbol: item.symbol,
        name: item.name,
        tracked,
        news: []
      };
    });
    
    return stocks;
  } catch (error) {
    console.error('Error creating mock stocks:', error);
    // Return empty array as fallback
    return [];
  }
};

// Initial mock data (will be used as fallback when CSV data cannot be loaded)
export const mockStocks: Stock[] = [
  {
    id: '1',
    symbol: 'ASELS',
    name: 'Aselsan Elektronik Sanayi ve Ticaret A.Ş.',
    tracked: true,
    news: []
  },
  {
    id: '2',
    symbol: 'GARAN',
    name: 'Türkiye Garanti Bankası A.Ş.',
    tracked: true,
    news: []
  },
  {
    id: '3',
    symbol: 'THYAO',
    name: 'Türk Hava Yolları A.O.',
    tracked: true,
    news: []
  },
  {
    id: '4',
    symbol: 'KCHOL',
    name: 'Koç Holding A.Ş.',
    tracked: false,
    news: []
  },
  {
    id: '5',
    symbol: 'EREGL',
    name: 'Ereğli Demir ve Çelik Fabrikaları T.A.Ş.',
    tracked: false,
    news: []
  }
];

export const mockEmailSettings: EmailSettings = {
  enabled: false,
  address: '',
  frequency: 'daily',
  signalThreshold: 'strong'
};
