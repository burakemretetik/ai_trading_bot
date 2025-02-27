
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Header from '@/components/Header';
import SearchBar from '@/components/SearchBar';
import StockCard from '@/components/StockCard';
import EmptyState from '@/components/EmptyState';
import EmailSetup from '@/components/EmailSetup';
import { Stock, EmailSettings } from '@/utils/types';
import { mockStocks, mockEmailSettings } from '@/utils/mockData';
import { Button } from '@/components/ui/button';
import { List } from 'lucide-react';
import { toast } from 'sonner';

const Index = () => {
  const [stocks, setStocks] = useState<Stock[]>(mockStocks);
  const [trackedStocks, setTrackedStocks] = useState<Stock[]>([]);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isEmailSetupOpen, setIsEmailSetupOpen] = useState(false);
  const [emailSettings, setEmailSettings] = useState<EmailSettings>(mockEmailSettings);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API loading
    const timer = setTimeout(() => {
      const filtered = stocks.filter(stock => stock.tracked);
      setTrackedStocks(filtered);
      setLoading(false);
    }, 800);
    
    return () => clearTimeout(timer);
  }, [stocks]);

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

  return (
    <div className="min-h-screen bg-background">
      <Header 
        onSearchClick={() => setIsSearchOpen(true)} 
        onSettingsClick={() => setIsEmailSetupOpen(true)} 
      />
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 animate-pulse">
            <div className="h-8 w-48 bg-muted rounded-lg mb-8"></div>
            <div className="h-64 w-full max-w-2xl bg-muted rounded-lg"></div>
          </div>
        ) : trackedStocks.length > 0 ? (
          <>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-semibold animate-fade-in">Takip Ettiğiniz Hisseler</h2>
              <Button variant="outline" size="sm" asChild>
                <Link to="/stocks" className="flex items-center gap-2">
                  <List className="h-4 w-4" />
                  <span>Tüm Hisseler</span>
                </Link>
              </Button>
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
