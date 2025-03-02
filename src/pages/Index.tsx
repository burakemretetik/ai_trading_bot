
import React, { useState } from 'react';
import Header from '@/components/Header';
import IndexHeader from '@/components/IndexHeader';
import TrackedStocksList from '@/components/TrackedStocksList';
import SearchBar from '@/components/SearchBar';
import WhatsAppSettings from '@/components/WhatsAppSettings';
import { useStocks } from '@/hooks/useStocks';

export default function Index() {
  const { stocks, loading, handleToggleTracking, handleAddStock } = useStocks();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const handleSearchClick = () => {
    setIsSearchOpen(true);
  };

  const handleSettingsClick = () => {
    setShowSettings(!showSettings);
  };
  
  return (
    <div className="min-h-screen bg-background">
      <Header 
        onSearchClick={handleSearchClick}
        onSettingsClick={handleSettingsClick}
      />
      
      <main className="container mx-auto px-4 py-8">
        <IndexHeader onAddStock={handleSearchClick} />
        
        {showSettings ? (
          <div className="my-6">
            <WhatsAppSettings />
          </div>
        ) : (
          <TrackedStocksList 
            stocks={stocks}
            loading={loading}
            onToggleTracking={handleToggleTracking}
            onSearchClick={handleSearchClick}
          />
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
