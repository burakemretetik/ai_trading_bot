
import { Stock, EmailSettings } from './types';

// Base mock news data that we'll reuse
const mockNewsItems = [
  {
    id: '101',
    title: "Yeni Stratejik Yatırım Duyuruldu",
    source: 'BloombergHT',
    url: '#',
    publishedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
    summary: "Şirket yönetimi yeni stratejik yatırımı düzenlenen basın toplantısında duyurdu. Bu yatırım, şirketin büyüme hedeflerine önemli katkı sağlayacak."
  },
  {
    id: '201',
    title: "Üçüncü Çeyrek Finansal Sonuçlar Beklentilerin Üzerinde",
    source: 'Ekonomist',
    url: '#',
    publishedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    summary: 'Şirketin üçüncü çeyrek finansal sonuçları analist beklentilerinin üzerinde gerçekleşti. Hisse senedi değer kazanıyor.'
  },
  {
    id: '301',
    title: 'Yeni Teknoloji Yatırımı Açıklandı',
    source: 'Habertürk',
    url: '#',
    publishedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
    summary: "Şirket, yeni teknoloji yatırımlarını duyurdu. Bu yatırım, şirketin rekabet gücünü artıracak ve operasyonel verimliliğe katkı sağlayacak."
  },
  {
    id: '401',
    title: 'Yönetim Kurulunda Değişiklik',
    source: 'Dünya',
    url: '#',
    publishedAt: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
    summary: 'Şirketin yönetim kurulunda önemli değişiklikler yaşandı. Yeni atamalar şirketin stratejik hedeflerine ulaşmasına katkı sağlayacak.'
  },
  {
    id: '501',
    title: 'Üretim Kapasitesi Artırılıyor',
    source: 'Anadolu Ajansı',
    url: '#',
    publishedAt: new Date(Date.now() - 10 * 60 * 60 * 1000).toISOString(),
    summary: 'Şirket, üretim kapasitesini artırmak için yeni yatırım projesini hayata geçireceğini duyurdu.'
  }
];

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
    
    // Create mock stocks from CSV data (taking first 50 for better performance)
    const stocks: Stock[] = csvData.slice(0, 50).map((item, index) => {
      // Assign a random tracked status (with some being tracked by default)
      const tracked = index < 5 || Math.random() > 0.8;
      
      // Assign random news items to some stocks
      const hasNews = Math.random() > 0.6;
      const news = hasNews 
        ? [{
            ...mockNewsItems[Math.floor(Math.random() * mockNewsItems.length)]
          }]
        : [];
      
      return {
        id: (index + 1).toString(),
        symbol: item.symbol,
        name: item.name,
        tracked,
        news
      };
    });
    
    return stocks;
  } catch (error) {
    console.error('Error creating mock stocks:', error);
    // Return the original mock data as fallback
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
    news: [
      {
        id: '101',
        title: "Aselsan Yeni Savunma Sistemini Duyurdu",
        source: 'BloombergHT',
        url: '#',
        publishedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        summary: "Aselsan, yeni geliştirdiği savunma sistemini düzenlenen basın toplantısında duyurdu. Sistem, Türkiye'nin savunma yeteneğini önemli ölçüde artıracak."
      }
    ]
  },
  {
    id: '2',
    symbol: 'GARAN',
    name: 'Türkiye Garanti Bankası A.Ş.',
    tracked: true,
    news: [
      {
        id: '201',
        title: "Garanti BBVA'nın Dijital Bankacılık Kullanıcı Sayısı 12 Milyonu Aştı",
        source: 'Ekonomist',
        url: '#',
        publishedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
        summary: 'Türkiye Garanti Bankası, dijital bankacılık kullanıcı sayısının 12 milyonu aştığını ve mobil işlemlerde rekor kırıldığını açıkladı.'
      }
    ]
  },
  {
    id: '3',
    symbol: 'THYAO',
    name: 'Türk Hava Yolları A.O.',
    tracked: true,
    news: [
      {
        id: '301',
        title: 'THY, Yeni Uçak Siparişlerini Açıkladı',
        source: 'Habertürk',
        url: '#',
        publishedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
        summary: "Türk Hava Yolları, filosunu genişletmek için yeni uçak siparişlerini duyurdu. Bu yatırım, şirketin büyüme stratejisinin önemli bir adımı olarak görülüyor."
      }
    ]
  },
  {
    id: '4',
    symbol: 'KCHOL',
    name: 'Koç Holding A.Ş.',
    tracked: false,
    news: [
      {
        id: '401',
        title: 'Koç Holding Yeni Yatırım Planlarını Açıkladı',
        source: 'Dünya',
        url: '#',
        publishedAt: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
        summary: 'Koç Holding, önümüzdeki beş yıl için yeni yatırım planlarını ve stratejik hedeflerini açıkladı.'
      }
    ]
  },
  {
    id: '5',
    symbol: 'EREGL',
    name: 'Ereğli Demir ve Çelik Fabrikaları T.A.Ş.',
    tracked: false,
    news: [
      {
        id: '501',
        title: 'Erdemir Üretim Kapasitesini Artırıyor',
        source: 'Anadolu Ajansı',
        url: '#',
        publishedAt: new Date(Date.now() - 10 * 60 * 60 * 1000).toISOString(),
        summary: 'Ereğli Demir ve Çelik Fabrikaları, üretim kapasitesini artırmak için yeni yatırım projesini hayata geçireceğini duyurdu.'
      }
    ]
  }
];

export const mockEmailSettings: EmailSettings = {
  enabled: false,
  address: '',
  frequency: 'daily',
  signalThreshold: 'strong'
};
