
import { Stock, EmailSettings } from './types';

// Adding Turkish stocks from Borsa İstanbul
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
        summary: "Aselsan, yeni geliştirdiği savunma sistemini düzenlenen basın toplantısında duyurdu. Sistem, Türkiye'nin savunma yeteneğini önemli ölçüde artıracak.",
        signalStrength: 'strong'
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
        summary: 'Türkiye Garanti Bankası, dijital bankacılık kullanıcı sayısının 12 milyonu aştığını ve mobil işlemlerde rekor kırıldığını açıkladı.',
        signalStrength: 'medium'
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
        summary: "Türk Hava Yolları, filosunu genişletmek için yeni uçak siparişlerini duyurdu. Bu yatırım, şirketin büyüme stratejisinin önemli bir adımı olarak görülüyor.",
        signalStrength: 'strong'
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
        summary: 'Koç Holding, önümüzdeki beş yıl için yeni yatırım planlarını ve stratejik hedeflerini açıkladı.',
        signalStrength: 'neutral'
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
        summary: 'Ereğli Demir ve Çelik Fabrikaları, üretim kapasitesini artırmak için yeni yatırım projesini hayata geçireceğini duyurdu.',
        signalStrength: 'medium'
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
