
import { Stock, EmailSettings } from './types';

// Adding Turkish stocks from Borsa İstanbul
export const mockStocks: Stock[] = [
  {
    id: '1',
    symbol: 'ASELS',
    name: 'Aselsan Elektronik Sanayi ve Ticaret A.Ş.',
    price: 52.80,
    priceChange: 1.25,
    tracked: true,
    news: [
      {
        id: '101',
        title: 'Aselsan Yeni Savunma Sistemini Duyurdu',
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
    price: 36.42,
    priceChange: 0.54,
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
    price: 241.50,
    priceChange: 3.75,
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
    price: 104.80,
    priceChange: -1.20,
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
    price: 37.16,
    priceChange: -0.48,
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
  },
  {
    id: '6',
    symbol: 'ISCTR',
    name: 'Türkiye İş Bankası A.Ş.',
    price: 12.45,
    priceChange: 0.22,
    tracked: false,
    news: [
      {
        id: '601',
        title: 'İş Bankası Yeni Kredi Paketini Açıkladı',
        source: 'Finans Gündem',
        url: '#',
        publishedAt: new Date(Date.now() - 9 * 60 * 60 * 1000).toISOString(),
        summary: "İş Bankası, KOBİ'lere yönelik yeni kredi paketini duyurdu. Paket, düşük faizli ve uzun vadeli kredi imkanları sunuyor.",
        signalStrength: 'medium'
      }
    ]
  },
  {
    id: '7',
    symbol: 'SISE',
    name: 'Türkiye Şişe ve Cam Fabrikaları A.Ş.',
    price: 24.68,
    priceChange: 0.34,
    tracked: false,
    news: [
      {
        id: '701',
        title: 'Şişecam İhracat Rakamlarında Rekor Kırdı',
        source: 'Ekonomi Haberleri',
        url: '#',
        publishedAt: new Date(Date.now() - 11 * 60 * 60 * 1000).toISOString(),
        summary: 'Türkiye Şişe ve Cam Fabrikaları, geçen yıla göre ihracat rakamlarında %25 artış sağlayarak yeni bir rekora imza attı.',
        signalStrength: 'weak'
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
