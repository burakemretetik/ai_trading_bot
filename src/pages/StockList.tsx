
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Star, Search, CheckSquare, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Stock } from '@/utils/types';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { toast } from 'sonner';
import { getTrackedStocks, trackStock, untrackStock } from '@/services/stockService';
import { Switch } from '@/components/ui/switch';

const StockList = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);
  const [isFollowAllActive, setIsFollowAllActive] = useState(false);

  useEffect(() => {
    const loadStocks = async () => {
      setLoading(true);
      try {
        // Try to load stocks from CSV
        const csvStocks = await createMockStocksFromCSV();

        // Get tracked stocks from localStorage
        const trackedStockIds = await getTrackedStocks();
        console.log('Tracked stock IDs:', trackedStockIds);

        // Use CSV data if available, otherwise fall back to mock data
        const baseStocks = csvStocks && csvStocks.length > 0 ? csvStocks : mockStocks;

        // Update tracking status based on localStorage data
        const updatedStocks = baseStocks.map(stock => ({
          ...stock,
          tracked: trackedStockIds.includes(stock.id)
        }));
        
        setStocks(updatedStocks);
        
        // Check if all stocks are being tracked to set the toggle state
        const allTracked = updatedStocks.length > 0 && 
          updatedStocks.every(stock => stock.tracked);
        setIsFollowAllActive(allTracked);
        
      } catch (error) {
        console.error('Error loading stocks:', error);

        // For error fallback, also check localStorage
        const trackedStockIds = await getTrackedStocks();
        const updatedMockStocks = mockStocks.map(stock => ({
          ...stock,
          tracked: trackedStockIds.includes(stock.id)
        }));
        setStocks(updatedMockStocks);
        
        // Check if all stocks are being tracked in mock data
        const allTracked = updatedMockStocks.length > 0 && 
          updatedMockStocks.every(stock => stock.tracked);
        setIsFollowAllActive(allTracked);
        
        toast.error('Hisse verileri yüklenirken bir hata oluştu');
      } finally {
        setLoading(false);
      }
    };
    loadStocks();
  }, []);

  useEffect(() => {
    // Filter stocks based on search query
    if (searchQuery.trim() === '') {
      setFilteredStocks(stocks);
    } else {
      const query = searchQuery.toLowerCase();
      setFilteredStocks(stocks.filter(stock => stock.symbol.toLowerCase().includes(query) || stock.name.toLowerCase().includes(query)));
    }
  }, [searchQuery, stocks]);

  const handleToggleTracking = async (id: string) => {
    const stockToUpdate = stocks.find(stock => stock.id === id);
    if (!stockToUpdate) return;
    setStocks(prev => prev.map(s => {
      if (s.id === id) {
        return {
          ...s,
          tracked: !s.tracked
        };
      }
      return s;
    }));
    
    try {
      if (!stockToUpdate.tracked) {
        // Track the stock
        const success = await trackStock(id);
        if (success) {
          toast.success(`${stockToUpdate.symbol} hisse takibinize eklendi`);
        }
      } else {
        // Untrack the stock
        const success = await untrackStock(id);
        if (success) {
          toast(`${stockToUpdate.symbol} hisse takibinizden çıkarıldı`);
        }
      }
      
      // Update the follow all toggle based on current tracking state
      const allTracked = stocks.every(stock => 
        stock.id === id ? !stockToUpdate.tracked : stock.tracked
      );
      setIsFollowAllActive(allTracked);
      
    } catch (error) {
      console.error('Error toggling stock tracking:', error);

      // Revert the UI change if the API call failed
      setStocks(prevStocks => prevStocks.map(stock => {
        if (stock.id === id) {
          return {
            ...stock,
            tracked: stockToUpdate.tracked
          };
        }
        return stock;
      }));
      toast.error('İşlem sırasında bir hata oluştu');
    }
  };
  
  const handleToggleFollowAll = async () => {
    const newFollowAllState = !isFollowAllActive;
    setIsFollowAllActive(newFollowAllState);
    
    // First update the UI for immediate feedback
    setStocks(prev => prev.map(stock => ({
      ...stock,
      tracked: newFollowAllState
    })));
    
    // Then process the changes in the background
    let successCount = 0;
    let failCount = 0;
    
    try {
      // Create an array of promises for all the tracking operations
      const operations = stocks.map(async (stock) => {
        try {
          if (newFollowAllState && !stock.tracked) {
            // Track the stock if it's not already tracked
            const success = await trackStock(stock.id);
            if (success) successCount++;
            else failCount++;
          } else if (!newFollowAllState && stock.tracked) {
            // Untrack the stock if it's currently tracked
            const success = await untrackStock(stock.id);
            if (success) successCount++;
            else failCount++;
          }
        } catch (error) {
          console.error(`Error processing stock ${stock.symbol}:`, error);
          failCount++;
        }
      });
      
      // Wait for all operations to complete
      await Promise.all(operations);
      
      // Show success message
      if (successCount > 0) {
        if (newFollowAllState) {
          toast.success(`${successCount} hisse takibinize eklendi`);
        } else {
          toast(`${successCount} hisse takibinizden çıkarıldı`);
        }
      }
      
      // Show error message if any operations failed
      if (failCount > 0) {
        toast.error(`${failCount} hisse işlemi başarısız oldu`);
      }
    } catch (error) {
      console.error('Error in bulk operation:', error);
      toast.error('İşlem sırasında bir hata oluştu');
      
      // If the overall operation fails, reload the current state from localStorage
      const trackedStockIds = await getTrackedStocks();
      setStocks(prev => prev.map(stock => ({
        ...stock,
        tracked: trackedStockIds.includes(stock.id)
      })));
      
      // Update follow all toggle based on reloaded state
      const reloadedAllTracked = stocks.length > 0 && 
        stocks.every(stock => trackedStockIds.includes(stock.id));
      setIsFollowAllActive(reloadedAllTracked);
    }
  };

  return <div className="min-h-screen bg-background">
      <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" asChild>
              <Link to="/">
                <ArrowLeft className="h-5 w-5" />
              </Link>
            </Button>
            <h1 className="text-xl font-semibold">BIST 100 Hisseleri</h1>
          </div>
          
          <Button variant="ghost" size="icon" asChild>
            <Link to="/">
              <Settings className="h-5 w-5" />
            </Link>
          </Button>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {loading ? <div className="animate-pulse space-y-4">
            {[...Array(10)].map((_, i) => <div key={i} className="h-20 bg-muted rounded-lg"></div>)}
          </div> : <div className="space-y-4">
            <div className="flex justify-between mb-6">
              <h2 className="text-lg font-medium">Takip edebileceğiniz hisseler</h2>
              <div className="text-sm text-muted-foreground">
                {stocks.filter(s => s.tracked).length} takip edilen hisse
              </div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-card rounded-lg border mb-4">
              <div className="flex items-center gap-2">
                <CheckSquare className="h-5 w-5 text-primary" />
                <span className="font-medium">Tüm hisseleri takip et</span>
              </div>
              <Switch 
                checked={isFollowAllActive} 
                onCheckedChange={handleToggleFollowAll}
                aria-label="Tüm hisseleri takip et"
              />
            </div>
            
            <div className="relative mb-6">
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <Search className="h-4 w-4 text-muted-foreground" />
              </div>
              <Input type="text" placeholder="Hisse adı veya sembol ara..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)} className="pl-10" />
            </div>
            
            <div className="bg-card rounded-lg border overflow-hidden">
              <div className="grid grid-cols-12 px-4 py-3 bg-muted/50 border-b text-sm font-medium">
                <div className="col-span-3">Sembol</div>
                <div className="col-span-7">Şirket Adı</div>
                <div className="col-span-2 text-center">Takip</div>
              </div>
              
              {filteredStocks.length > 0 ? filteredStocks.map(stock => <div key={stock.id} className="grid grid-cols-12 px-4 py-4 border-b last:border-b-0 items-center hover:bg-muted/30 transition-colors">
                    <div className="col-span-3 font-medium">{stock.symbol}</div>
                    <div className="col-span-7 text-sm truncate" title={stock.name}>
                      {stock.name}
                    </div>
                    <div className="col-span-2 flex justify-center">
                      <Button variant="ghost" size="icon" className="rounded-full" onClick={() => handleToggleTracking(stock.id)}>
                        <Star className={`h-5 w-5 ${stock.tracked ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'}`} />
                        <span className="sr-only">
                          {stock.tracked ? 'Takipten çıkar' : 'Takip et'}
                        </span>
                      </Button>
                    </div>
                  </div>) : <div className="py-8 text-center">
                  <p className="text-muted-foreground">"{searchQuery}" için sonuç bulunamadı</p>
                </div>}
            </div>
          </div>}
      </main>
    </div>;
};
export default StockList;
