
import React from 'react';
import { NewsItem } from '@/utils/types';

interface StockNewsProps {
  news: NewsItem[];
  newsUrls: string[];
}

const StockNews: React.FC<StockNewsProps> = ({ news, newsUrls }) => {
  return <div className="flex-grow"></div>;
};

export default StockNews;
