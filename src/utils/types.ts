
export interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  publishedAt: string;
  summary: string;
  sentiment?: "positive" | "negative" | "neutral";
  signalStrength?: SignalStrength;
}

export interface Stock {
  id: string;
  symbol: string;
  name: string;
  tracked: boolean;
  news: NewsItem[];
}

export interface EmailSettings {
  enabled: boolean;
  address: string;
  frequency: "instant" | "daily" | "weekly";
  signalThreshold: string;
}

export type SignalStrength = "strong" | "medium" | "weak" | "neutral";

export interface StockNewsMapping {
  timestamp: string;
  updated: boolean;
  stock_news: {
    [stockName: string]: string[];
  };
}
