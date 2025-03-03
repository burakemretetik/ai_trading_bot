
import React, { useState } from 'react';
import Header from '@/components/Header';
import IndexHeader from '@/components/IndexHeader';
import TrackedStocksList from '@/components/TrackedStocksList';
import WhatsAppSettings from '@/components/WhatsAppSettings';
import SearchBar from '@/components/SearchBar';
import { useStocks } from '@/hooks/useStocks';

export default function Index() {
  const { stocks, loading, handleToggleTracking, handleAddStock, isFollowAllActive, handleToggleFollowAll } = useStocks();
  const [showSettings, setShowSettings] = useState(false);
  const [showSearchBar, setShowSearchBar] = useState(false);

  const handleSettingsClick = () => {
    setShowSettings(!showSettings);
  };
  
  const handleSearchClick = () => {
    setShowSearchBar(true);
  };
  
  const handleCloseSearch = () => {
    setShowSearchBar(false);
  };
  
  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)]">
      <Header 
        onSettingsClick={handleSettingsClick}
      />
      
      <main className="container mx-auto px-4 py-8 flex-grow">
        <IndexHeader />
        
        {showSettings ? (
          <div className="my-6">
            <WhatsAppSettings />
          </div>
        ) : (
          <div className="mt-6">
            <TrackedStocksList 
              stocks={stocks}
              loading={loading}
              onToggleTracking={handleToggleTracking}
              isFollowAllActive={isFollowAllActive}
              onToggleFollowAll={handleToggleFollowAll}
            />
          </div>
        )}
      </main>
      
      {/* Search Bar Modal */}
      <SearchBar
        isOpen={showSearchBar}
        onClose={handleCloseSearch}
        onAddStock={handleAddStock}
      />
    </div>
  );
}
