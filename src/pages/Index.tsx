
import React, { useState } from 'react';
import Header from '@/components/Header';
import IndexHeader from '@/components/IndexHeader';
import TrackedStocksList from '@/components/TrackedStocksList';
import SearchBar from '@/components/SearchBar';
import { useStocks } from '@/hooks/useStocks';
import { toast } from 'sonner';

export default function Index() {
  const { stocks, loading, handleToggleTracking, handleAddStock } = useStocks();
  const [isSearchOpen, setIsSearchOpen] = useState(false);

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
        <IndexHeader onAddStock={handleSearchClick} />
        
        <TrackedStocksList 
          stocks={stocks}
          loading={loading}
          onToggleTracking={handleToggleTracking}
          onSearchClick={handleSearchClick}
        />
      </main>
      
      <SearchBar
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onAddStock={handleAddStock}
      />
    </div>
  );
}
