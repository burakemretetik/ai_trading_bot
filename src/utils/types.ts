
export interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  publishedAt: string;
  summary: string;
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

export interface UserProfile {
  id: string;
  username: string | null;
  avatar_url: string | null;
}

export interface Session {
  user: UserProfile | null;
  isLoading: boolean;
}
