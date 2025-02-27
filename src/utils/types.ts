
export type SignalStrength = "strong" | "medium" | "weak" | "neutral";

export interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  publishedAt: string;
  summary: string;
  signalStrength: SignalStrength;
}

export interface Stock {
  id: string;
  symbol: string;
  name: string;
  price: number;
  priceChange: number;
  tracked: boolean;
  news: NewsItem[];
}

export interface EmailSettings {
  enabled: boolean;
  address: string;
  frequency: "instant" | "daily" | "weekly";
  signalThreshold: SignalStrength;
}
