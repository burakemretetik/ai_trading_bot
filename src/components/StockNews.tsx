
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
      <div className="space-y-2 mt-1">
        {news.slice(0, 3).map(item => (
          <NewsItemComponent key={item.id} news={item} />
        ))}
      </div>
    );
  } 
  
  if (newsUrls.length > 0) {
    return (
      <div className="space-y-1">
        {newsUrls.slice(0, 3).map((url, index) => (
          <a 
            key={index} 
            href={url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="block py-1 text-xs hover:text-primary transition-colors flex items-center text-muted-foreground"
          >
            <Globe className="h-3 w-3 mr-1" />
            <span className="truncate">{new URL(url).hostname.replace('www.', '')}</span>
          </a>
        ))}
      </div>
    );
  }
  
  return <div className="flex-grow"></div>;
};

export default StockNews;
