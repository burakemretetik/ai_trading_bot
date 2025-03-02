
import React from 'react';
import { Stock } from '@/utils/types';
import StockCard from '@/components/StockCard';
import EmptyState from '@/components/EmptyState';

type TrackedStocksListProps = {
  stocks: Stock[];
  loading: boolean;
  onToggleTracking: (id: string) => void;
  onSearchClick: () => void;
};

const TrackedStocksList = ({ 
  stocks, 
  loading, 
  onToggleTracking, 
  onSearchClick 
}: TrackedStocksListProps) => {
  const trackedStocks = stocks.filter(stock => stock.tracked);
  const hasNewsInTrackedStocks = trackedStocks.some(stock => stock.news.length > 0);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="border rounded-lg h-64 animate-pulse bg-muted"></div>
        ))}
      </div>
    );
  }

  if (trackedStocks.length === 0) {
    return <EmptyState onSearchClick={onSearchClick} />;
  }

  return (
    <>
      {!hasNewsInTrackedStocks && (
        <div className="p-8 text-center bg-muted rounded-lg mb-6">
          <p className="text-lg font-medium">Takip ettiğiniz hisselerle ilgili yeni bir gelişme yok.</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {trackedStocks.map(stock => (
          <StockCard
            key={stock.id}
            stock={stock}
            onToggleTracking={onToggleTracking}
          />
        ))}
      </div>
    </>
  );
};

export default TrackedStocksList;
