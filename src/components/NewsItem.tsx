import React from 'react';
import { ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { NewsItem as NewsItemType, SignalStrength } from '@/utils/types';
import { formatDistanceToNow } from 'date-fns';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
interface NewsItemProps {
  news: NewsItemType;
}
const NewsItem: React.FC<NewsItemProps> = ({
  news
}) => {
  const formattedDate = formatDistanceToNow(new Date(news.publishedAt), {
    addSuffix: true
  });
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
      neutral: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
    };
    return <Badge variant="outline" className={cn("ml-2 text-xs font-normal", variants[news.signalStrength])}>
        {news.signalStrength} signal
      </Badge>;
  };
  return;
};
export default NewsItem;