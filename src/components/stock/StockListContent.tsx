
import React from 'react';
import { Stock } from '@/utils/types';
import StockListItem from './StockListItem';

interface StockListContentProps {
  filteredStocks: Stock[];
  searchQuery: string;
  onToggleTracking: (id: string) => void;
}

const StockListContent: React.FC<StockListContentProps> = ({ 
  filteredStocks, 
  searchQuery, 
  onToggleTracking 
}) => {
  return (
    <div className="bg-card rounded-lg border overflow-hidden">
      <div className="grid grid-cols-12 px-4 py-3 bg-muted/50 border-b text-sm font-medium">
        <div className="col-span-3">Sembol</div>
        <div className="col-span-7">Şirket Adı</div>
        <div className="col-span-2 text-center">Takip</div>
      </div>
      
      {filteredStocks.length > 0 ? (
        filteredStocks.map(stock => (
          <StockListItem 
            key={stock.id} 
            stock={stock} 
            onToggleTracking={onToggleTracking} 
          />
        ))
      ) : (
        <div className="py-8 text-center">
          <p className="text-muted-foreground">"{searchQuery}" için sonuç bulunamadı</p>
        </div>
      )}
    </div>
  );
};

export default StockListContent;
