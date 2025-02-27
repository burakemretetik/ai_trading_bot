
import React from 'react';
import { ExternalLink } from 'lucide-react';
import { NewsItem as NewsItemType, SignalStrength } from '@/utils/types';
import { formatDistanceToNow } from 'date-fns';
import { cn } from '@/lib/utils';

interface NewsItemProps {
  news: NewsItemType;
}

const getSignalColor = (signal?: SignalStrength): string => {
  switch (signal) {
    case 'strong':
      return 'bg-green-100 text-green-800';
    case 'medium':
      return 'bg-yellow-100 text-yellow-800';
    case 'weak':
      return 'bg-orange-100 text-orange-800';
    case 'neutral':
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

const NewsItem: React.FC<NewsItemProps> = ({ news }) => {
  const formattedDate = formatDistanceToNow(new Date(news.publishedAt), { addSuffix: true });
  
  return (
    <div className="p-4 border rounded-lg bg-card card-hover mb-4 animate-slide-up">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium bg-secondary px-2 py-1 rounded-full">{news.source}</span>
          <span className="text-xs text-muted-foreground">{formattedDate}</span>
          
          {news.signalStrength && (
            <span className={cn(
              "text-xs font-medium px-2 py-1 rounded-full ml-2",
              getSignalColor(news.signalStrength)
            )}>
              {news.signalStrength.charAt(0).toUpperCase() + news.signalStrength.slice(1)}
            </span>
          )}
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
