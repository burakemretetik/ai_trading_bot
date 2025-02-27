
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Check, X, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Stock } from '@/utils/types';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { toast } from 'sonner';

const StockList = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStocks = async () => {
      setLoading(true);
      try {
        // Try to load stocks from CSV
        const csvStocks = await createMockStocksFromCSV();
        
        // Use CSV data if available, otherwise fall back to mock data
        if (csvStocks && csvStocks.length > 0) {
          setStocks(csvStocks);
        } else {
          setStocks(mockStocks);
        }
      } catch (error) {
        console.error('Error loading stocks:', error);
        setStocks(mockStocks);
        toast.error('Hisse verileri yüklenirken bir hata oluştu');
      } finally {
        setLoading(false);
      }
    };
    
    // Simulate API loading
    const timer = setTimeout(() => {
      loadStocks();
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);

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
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {loading ? (
          <div className="animate-pulse space-y-4">
            {[...Array(10)].map((_, i) => (
              <div key={i} className="h-20 bg-muted rounded-lg"></div>
            ))}
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
                <div className="col-span-3">Sembol</div>
                <div className="col-span-7">Şirket Adı</div>
                <div className="col-span-2 text-center">Takip</div>
              </div>
              
              {stocks.map(stock => (
                <div 
                  key={stock.id} 
                  className="grid grid-cols-12 px-4 py-4 border-b last:border-b-0 items-center hover:bg-muted/30 transition-colors"
                >
                  <div className="col-span-3 font-medium">{stock.symbol}</div>
                  <div className="col-span-7 text-sm truncate" title={stock.name}>
                    {stock.name}
                  </div>
                  <div className="col-span-2 flex justify-center">
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
