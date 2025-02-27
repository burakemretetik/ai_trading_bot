
import React from 'react';
import { ExternalLink } from 'lucide-react';
import { NewsItem as NewsItemType } from '@/utils/types';
import { formatDistanceToNow } from 'date-fns';

interface NewsItemProps {
  news: NewsItemType;
}

const NewsItem: React.FC<NewsItemProps> = ({ news }) => {
  const formattedDate = formatDistanceToNow(new Date(news.publishedAt), { addSuffix: true });
  
  // Check if this is a search result (they have a different visual treatment)
  const isSearchResult = news.id.startsWith('search-');
  
  return (
    <div className={`p-4 border rounded-lg bg-card ${isSearchResult ? 'border-primary/20' : ''} card-hover mb-4 animate-slide-up`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center">
          <span className={`text-xs font-medium ${isSearchResult ? 'bg-primary/10 text-primary' : 'bg-secondary'} px-2 py-1 rounded-full`}>
            {news.source}
          </span>
          <span className="text-xs text-muted-foreground ml-2">{formattedDate}</span>
        </div>
      </div>
      
      <h3 className="font-semibold text-base mb-2">{news.title}</h3>
      <p className="text-sm text-muted-foreground mb-3">{news.summary}</p>
      
      <div className="flex justify-end">
        <a 
          href={news.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-xs flex items-center text-primary hover:underline transition-all"
        >
          <span>Read full article</span>
          <ExternalLink className="h-3 w-3 ml-1" />
        </a>
      </div>
    </div>
  );
};

export default NewsItem;
