
import React from 'react';
import { Stock } from '@/utils/types';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import StockHeader from './StockHeader';
import StockNews from './StockNews';
import { useStockNews } from '@/hooks/useStockNews';

interface StockCardProps {
  stock: Stock;
  onToggleTracking: (id: string) => void;
}

const StockCard: React.FC<StockCardProps> = ({
  stock,
  onToggleTracking
}) => {
  const { news, newsUrls } = useStockNews(stock);

  return (
    <Card className="h-full flex flex-col bg-white dark:bg-card border-gray-100 dark:border-gray-800 shadow-sm hover:shadow-md transition-shadow">
      <CardHeader className="pb-2 pt-4 px-4">
        <StockHeader 
          symbol={stock.symbol}
          name={stock.name}
          tracked={stock.tracked}
          onToggleTracking={onToggleTracking}
          stockId={stock.id}
        />
      </CardHeader>
      
      <CardContent className="pt-1 px-4 flex-grow">
        <StockNews news={news} newsUrls={newsUrls} />
      </CardContent>
    </Card>
  );
};

export default StockCard;
