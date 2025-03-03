
import React from 'react';
import { Globe } from 'lucide-react';
import { NewsItem } from '@/utils/types';
import NewsItemComponent from './NewsItem';

interface StockNewsProps {
  news: NewsItem[];
  newsUrls: string[];
}

const StockNews: React.FC<StockNewsProps> = ({ news, newsUrls }) => {
  if (news.length > 0) {
    return (
      <div className="space-y-3 mt-2">
        {news.slice(0, 3).map(item => (
          <NewsItemComponent key={item.id} news={item} />
        ))}
      </div>
    );
  } 
  
  if (newsUrls.length > 0) {
    return (
      <div className="space-y-2">
        {newsUrls.slice(0, 3).map((url, index) => (
          <a 
            key={index} 
            href={url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="block p-2 border rounded hover:bg-muted transition-colors flex items-center"
          >
            <Globe className="h-4 w-4 mr-2" />
            <span className="text-sm truncate">{new URL(url).hostname.replace('www.', '')}</span>
          </a>
        ))}
      </div>
    );
  }
  
  return <div className="flex-grow"></div>;
};

export default StockNews;
