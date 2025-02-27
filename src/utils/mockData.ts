
import { Stock, EmailSettings } from './types';

export const mockStocks: Stock[] = [
  {
    id: '1',
    symbol: 'AAPL',
    name: 'Apple Inc.',
    price: 185.92,
    priceChange: 1.23,
    tracked: true,
    news: [
      {
        id: '101',
        title: 'Apple Announces New M3 MacBook Air with Breakthrough Performance',
        source: 'TechCrunch',
        url: '#',
        publishedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
        summary: "Apple today introduced the new MacBook Air with the powerful M3 chip, delivering even more performance and capabilities to the world's most popular laptop.",
        signalStrength: 'strong'
      },
      {
        id: '102',
        title: 'Apple Services Growth Continues to Impress Analysts',
        source: 'Bloomberg',
        url: '#',
        publishedAt: new Date(Date.now() - 18 * 60 * 60 * 1000).toISOString(),
        summary: "Apple's services segment continues to grow at a robust rate, potentially offsetting any slowdown in hardware sales according to analysts.",
        signalStrength: 'medium'
      }
    ]
  },
  {
    id: '2',
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    price: 420.55,
    priceChange: 2.89,
    tracked: true,
    news: [
      {
        id: '201',
        title: 'Microsoft Azure Revenue Surges 30% as Cloud Demand Accelerates',
        source: 'CNBC',
        url: '#',
        publishedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
        summary: 'Microsoft reported earnings that exceeded expectations, largely driven by continued momentum in its cloud computing division.',
        signalStrength: 'strong'
      }
    ]
  },
  {
    id: '3',
    symbol: 'GOOG',
    name: 'Alphabet Inc.',
    price: 175.22,
    priceChange: -0.52,
    tracked: true,
    news: [
      {
        id: '301',
        title: 'Google Faces New Antitrust Investigation',
        source: 'The Wall Street Journal',
        url: '#',
        publishedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
        summary: "Regulators announced a new probe into Google's advertising practices, citing potential monopolistic behavior.",
        signalStrength: 'weak'
      }
    ]
  },
  {
    id: '4',
    symbol: 'AMZN',
    name: 'Amazon.com, Inc.',
    price: 180.38,
    priceChange: 1.05,
    tracked: false,
    news: [
      {
        id: '401',
        title: 'Amazon Expands Same-Day Delivery to More Cities',
        source: 'Reuters',
        url: '#',
        publishedAt: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
        summary: 'Amazon is rolling out same-day delivery service to 15 additional metropolitan areas across the United States.',
        signalStrength: 'neutral'
      }
    ]
  },
  {
    id: '5',
    symbol: 'TSLA',
    name: 'Tesla, Inc.',
    price: 164.90,
    priceChange: -3.75,
    tracked: false,
    news: [
      {
        id: '501',
        title: 'Tesla Achieves Record Quarterly Deliveries Despite Supply Chain Challenges',
        source: 'Electrek',
        url: '#',
        publishedAt: new Date(Date.now() - 10 * 60 * 60 * 1000).toISOString(),
        summary: 'Tesla delivered over 450,000 vehicles in Q1, beating analyst expectations despite ongoing supply chain difficulties.',
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
