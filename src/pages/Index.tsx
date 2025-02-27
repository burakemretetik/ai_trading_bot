
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import Header from '@/components/Header';
import SearchBar from '@/components/SearchBar';
import StockCard from '@/components/StockCard';
import EmptyState from '@/components/EmptyState';
import EmailSetup from '@/components/EmailSetup';
import { Stock, EmailSettings } from '@/utils/types';
import { mockEmailSettings } from '@/utils/mockData';
import { Button } from '@/components/ui/button';
import { List, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';
import { stockApi } from '@/services/stockApi';

const Index = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isEmailSetupOpen, setIsEmailSetupOpen] = useState(false);
  const [emailSettings, setEmailSettings] = useState<EmailSettings>(mockEmailSettings);

  // React Query hook for fetching stocks data
  const { isLoading, isError, refetch } = useQuery({
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

  // Filter tracked stocks
  const trackedStocks = stocks.filter(stock => stock.tracked);

  const handleToggleTracking = (id: string) => {
    setStocks(prev => 
      prev.map(stock => {
        if (stock.id === id) {
          const newTrackedState = !stock.tracked;
          
          // Show toast notification
          if (newTrackedState) {
            toast.success(`${stock.symbol} takibinize eklendi`);
          } else {
            toast(`${stock.symbol} takibinizden çıkarıldı`);
          }
          
          return { ...stock, tracked: newTrackedState };
        }
        return stock;
      })
    );
  };

  const handleAddStock = (stock: Stock) => {
    // Check if stock already exists
    const existingStock = stocks.find(s => s.id === stock.id);
    
    if (existingStock && !existingStock.tracked) {
      handleToggleTracking(stock.id);
    } else if (!existingStock) {
      // Add new stock with tracked = true
      setStocks(prev => [...prev, { ...stock, tracked: true }]);
      toast.success(`${stock.symbol} takibinize eklendi`);
    }
  };

  const handleSaveEmailSettings = (settings: EmailSettings) => {
    setEmailSettings(settings);
    if (settings.enabled) {
      toast.success('E-posta bildirimleri etkinleştirildi');
    }
  };

  const handleRefresh = () => {
    toast.info('Veriler güncelleniyor...');
    refetch();
  };

  return (
    <div className="min-h-screen bg-background">
      <Header 
        onSearchClick={() => setIsSearchOpen(true)} 
        onSettingsClick={() => setIsEmailSetupOpen(true)} 
      />
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-20 animate-pulse">
            <div className="h-8 w-48 bg-muted rounded-lg mb-8"></div>
            <div className="h-64 w-full max-w-2xl bg-muted rounded-lg"></div>
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center justify-center py-12">
            <p className="text-lg text-center text-red-500 mb-4">
              Hisse verileri alınırken bir hata oluştu.
            </p>
            <Button onClick={handleRefresh} className="flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Tekrar Dene
            </Button>
          </div>
        ) : trackedStocks.length > 0 ? (
          <>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold animate-fade-in">Takip Ettiğiniz Hisseler</h2>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleRefresh} className="flex items-center gap-2">
                  <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  <span>Yenile</span>
                </Button>
                <Button variant="outline" size="sm" asChild>
                  <Link to="/stocks" className="flex items-center gap-2">
                    <List className="h-4 w-4" />
                    <span>Tüm Hisseler</span>
                  </Link>
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
          <EmptyState onSearchClick={() => setIsSearchOpen(true)} />
        )}
      </main>
      
      <SearchBar 
        isOpen={isSearchOpen} 
        onClose={() => setIsSearchOpen(false)} 
        onAddStock={handleAddStock} 
      />
      
      <EmailSetup 
        isOpen={isEmailSetupOpen} 
        onClose={() => setIsEmailSetupOpen(false)} 
        settings={emailSettings} 
        onSave={handleSaveEmailSettings} 
      />
    </div>
  );
};

export default Index;
