
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
    
    // Add a small delay to simulate loading from an API
    const timer = setTimeout(() => {
      loadStocks();
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);
  
  const trackedStocks = stocks.filter(stock => stock.tracked);
  
  const handleToggleTracking = (id: string) => {
    setStocks(prevStocks => 
      prevStocks.map(stock => {
        if (stock.id === id) {
          const newTrackedState = !stock.tracked;
          
          if (newTrackedState) {
            toast.success(`${stock.symbol} hisse takibinize eklendi`);
          } else {
            toast(`${stock.symbol} hisse takibinizden çıkarıldı`);
          }
          
          return { ...stock, tracked: newTrackedState };
        }
        return stock;
      })
    );
  };
  
  const handleAddStock = (stock: Stock) => {
    if (!stocks.some(s => s.id === stock.id)) {
      setStocks(prev => [...prev, stock]);
    }
    
    handleToggleTracking(stock.id);
  };
  
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trackedStocks.map(stock => (
              <StockCard
                key={stock.id}
                stock={stock}
                onToggleTracking={handleToggleTracking}
              />
            ))}
          </div>
        ) : (
          <EmptyState />
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
