
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Plus, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { Stock } from '@/utils/types';
import EmptyState from '@/components/EmptyState';
import Header from '@/components/Header';
import StockCard from '@/components/StockCard';
import SearchBar from '@/components/SearchBar';
import { toast } from 'sonner';
import { getTrackedStocks, trackStock, untrackStock } from '@/services/stockService';
import { checkForNewsAndNotifyUser } from '@/services/newsService';

export default function Index() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  
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
        if (csvStocks && csvStocks.length > 0) {
          // Update tracking status based on localStorage data
          const updatedStocks = csvStocks.map(stock => ({
            ...stock,
            tracked: trackedStockIds.includes(stock.id)
          }));
          
          setStocks(updatedStocks);
        } else {
          // For mock data, also check localStorage
          const updatedMockStocks = mockStocks.map(stock => ({
            ...stock,
            tracked: trackedStockIds.includes(stock.id)
          }));
          
          setStocks(updatedMockStocks);
        }
        
        // Check for news updates
        await checkForNewsAndNotifyUser();
      } catch (error) {
        console.error('Error loading stocks:', error);
        
        // For error fallback, also check localStorage
        const trackedStockIds = await getTrackedStocks();
        
        const updatedMockStocks = mockStocks.map(stock => ({
          ...stock,
          tracked: trackedStockIds.includes(stock.id)
        }));
        
        setStocks(updatedMockStocks);
        toast.error('Hisse verileri yüklenirken bir hata oluştu');
      } finally {
        setLoading(false);
      }
    };
    
    loadStocks();
  }, []);
  
  const trackedStocks = stocks.filter(stock => stock.tracked);
  const hasNewsInTrackedStocks = trackedStocks.some(stock => stock.news.length > 0);
  
  const handleToggleTracking = async (id: string) => {
    const stockToUpdate = stocks.find(stock => stock.id === id);
    if (!stockToUpdate) return;
    
    setStocks(prevStocks => 
      prevStocks.map(stock => {
        if (stock.id === id) {
          const newTrackedState = !stock.tracked;
          return { ...stock, tracked: newTrackedState };
        }
        return stock;
      })
    );
    
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
    } catch (error) {
      console.error('Error toggling stock tracking:', error);
      
      // Revert the UI change if the API call failed
      setStocks(prevStocks => 
        prevStocks.map(stock => {
          if (stock.id === id) {
            return { ...stock, tracked: stockToUpdate.tracked };
          }
          return stock;
        })
      );
      
      toast.error('İşlem sırasında bir hata oluştu');
    }
  };
  
  const handleAddStock = async (stock: Stock) => {
    if (!stocks.some(s => s.id === stock.id)) {
      setStocks(prev => [...prev, stock]);
    }
    
    handleToggleTracking(stock.id);
  };

  const handleSearchClick = () => {
    setIsSearchOpen(true);
  };

  const handleSettingsClick = () => {
    // This would be implemented when settings functionality is needed
    toast('Settings functionality will be added soon');
  };
  
  return (
    <div className="min-h-screen bg-background">
      <Header 
        onSearchClick={handleSearchClick}
        onSettingsClick={handleSettingsClick}
      />
      
      <main className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Takip Ettiğiniz Hisseler</h1>
          
          <div className="flex space-x-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setIsSearchOpen(true)}
            >
              <Plus className="h-4 w-4 mr-2" />
              Hisse Ekle
            </Button>
            
            <Button variant="ghost" size="sm" asChild>
              <Link to="/stocks" className="flex items-center">
                Tüm Hisseler
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
        
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="border rounded-lg h-64 animate-pulse bg-muted"></div>
            ))}
          </div>
        ) : trackedStocks.length > 0 ? (
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
                  onToggleTracking={handleToggleTracking}
                />
              ))}
            </div>
          </>
        ) : (
          <EmptyState onSearchClick={handleSearchClick} />
        )}
      </main>
      
      <SearchBar
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onAddStock={handleAddStock}
      />
    </div>
  );
}
