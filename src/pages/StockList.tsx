
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Check, X, ArrowLeft, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Stock } from '@/utils/types';
import { mockStocks } from '@/utils/mockData';
import { toast } from 'sonner';
import { stockApi } from '@/services/stockApi';
import { useQuery } from '@tanstack/react-query';

const StockList = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);

  // React Query hook to fetch data
  const { isLoading, isError, data, refetch } = useQuery({
    queryKey: ['stocks'],
    queryFn: async () => {
      const response = await stockApi.getAllStocks();
      if (!response.success) {
        throw new Error(response.error);
      }
      return response.data;
    },
    onSuccess: (data) => {
      if (data) {
        // Set tracked status based on current state to maintain user selections
        const updatedStocks = data.map(newStock => {
          const existingStock = stocks.find(s => s.id === newStock.id);
          return {
            ...newStock,
            tracked: existingStock ? existingStock.tracked : newStock.tracked
          };
        });
        setStocks(updatedStocks);
      }
    },
    onError: (error) => {
      console.error('Hisse verileri alınırken hata oluştu:', error);
      toast.error('Veriler alınamadı. Lütfen daha sonra tekrar deneyin.');
    }
  });

  const handleToggleTracking = (stock: Stock) => {
    setStocks(prev => 
      prev.map(s => {
        if (s.id === stock.id) {
          const newTrackedState = !s.tracked;
          
          // Show toast notification
          if (newTrackedState) {
            toast.success(`${s.symbol} hisse takibinize eklendi`);
          } else {
            toast(`${s.symbol} hisse takibinizden çıkarıldı`);
          }
          
          return { ...s, tracked: newTrackedState };
        }
        return s;
      })
    );
  };

  const handleRefresh = () => {
    toast.info('Veriler güncelleniyor...');
    refetch();
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" asChild>
              <Link to="/">
                <ArrowLeft className="h-5 w-5" />
              </Link>
            </Button>
            <h1 className="text-xl font-semibold">Borsa İstanbul Hisseleri</h1>
          </div>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center gap-1"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Yenile</span>
          </Button>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {isLoading ? (
          <div className="animate-pulse space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-20 bg-muted rounded-lg"></div>
            ))}
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center justify-center py-12">
            <p className="text-lg text-center text-red-500 mb-4">
              Hisse verileri alınırken bir hata oluştu.
            </p>
            <Button onClick={() => refetch()} className="flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Tekrar Dene
            </Button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-between mb-6">
              <h2 className="text-lg font-medium">Takip edebileceğiniz hisseler</h2>
              <div className="text-sm text-muted-foreground">
                {stocks.filter(s => s.tracked).length} takip edilen hisse
              </div>
            </div>
            
            <div className="bg-card rounded-lg border overflow-hidden">
              <div className="grid grid-cols-12 px-4 py-3 bg-muted/50 border-b text-sm font-medium">
                <div className="col-span-2">Sembol</div>
                <div className="col-span-5">Şirket Adı</div>
                <div className="col-span-2 text-right">Fiyat</div>
                <div className="col-span-2 text-right">Değişim</div>
                <div className="col-span-1 text-center">Takip</div>
              </div>
              
              {stocks.map(stock => (
                <div 
                  key={stock.id} 
                  className="grid grid-cols-12 px-4 py-4 border-b last:border-b-0 items-center hover:bg-muted/30 transition-colors"
                >
                  <div className="col-span-2 font-medium">{stock.symbol}</div>
                  <div className="col-span-5 text-sm truncate" title={stock.name}>
                    {stock.name}
                  </div>
                  <div className="col-span-2 text-right font-medium">
                    {stock.price.toFixed(2)} ₺
                  </div>
                  <div className={`col-span-2 text-right font-medium flex items-center justify-end ${stock.priceChange >= 0 ? 'text-signal-strong' : 'text-signal-weak'}`}>
                    {stock.priceChange > 0 ? '+' : ''}{stock.priceChange.toFixed(2)} ₺
                  </div>
                  <div className="col-span-1 flex justify-center">
                    <Button 
                      variant={stock.tracked ? "default" : "outline"} 
                      size="sm"
                      className={`w-7 h-7 p-0 rounded-full ${stock.tracked ? 'bg-primary hover:bg-primary/90' : ''}`}
                      onClick={() => handleToggleTracking(stock)}
                    >
                      {stock.tracked ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <X className="h-4 w-4" />
                      )}
                      <span className="sr-only">
                        {stock.tracked ? 'Takipten çıkar' : 'Takip et'}
                      </span>
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default StockList;
