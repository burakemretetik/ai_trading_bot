
import React from 'react';
import { ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { NewsItem as NewsItemType, SignalStrength } from '@/utils/types';
import { formatDistanceToNow } from 'date-fns';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface NewsItemProps {
  news: NewsItemType;
}

const NewsItem: React.FC<NewsItemProps> = ({ news }) => {
  const formattedDate = formatDistanceToNow(new Date(news.publishedAt), { addSuffix: true });
  
  const getSentimentIcon = () => {
    if (!news.sentiment) return null;
    
    switch (news.sentiment) {
      case 'positive':
        return <TrendingUp className="h-3 w-3 text-green-500" />;
      case 'negative':
        return <TrendingDown className="h-3 w-3 text-red-500" />;
      case 'neutral':
        return <Minus className="h-3 w-3 text-gray-500" />;
      default:
        return null;
    }
  };
  
  const getSignalBadge = () => {
    if (!news.signalStrength) return null;
    
    const variants: Record<SignalStrength, string> = {
      strong: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
      medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300',
      weak: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300',
      neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
    };
    
    return (
      <Badge 
        variant="outline" 
        className={cn("ml-2 text-xs font-normal", variants[news.signalStrength])}
      >
        {news.signalStrength} signal
      </Badge>
    );
  };
  
  return (
    <div className="p-4 border rounded-lg bg-card card-hover mb-4 animate-slide-up">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center">
          <span className="text-xs font-medium bg-secondary px-2 py-1 rounded-full">{news.source}</span>
          <span className="text-xs text-muted-foreground ml-2">{formattedDate}</span>
          {news.sentiment && (
            <div className="flex items-center ml-2">
              {getSentimentIcon()}
            </div>
          )}
          {getSignalBadge()}
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
