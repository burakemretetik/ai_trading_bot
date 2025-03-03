
import React, { useState } from 'react';
import Header from '@/components/Header';
import IndexHeader from '@/components/IndexHeader';
import TrackedStocksList from '@/components/TrackedStocksList';
import WhatsAppSettings from '@/components/WhatsAppSettings';
import { useStocks } from '@/hooks/useStocks';

export default function Index() {
  const { stocks, loading, handleToggleTracking, handleAddStock } = useStocks();
  const [showSettings, setShowSettings] = useState(false);

  const handleSettingsClick = () => {
    setShowSettings(!showSettings);
  };
  
  return (
    <div className="min-h-screen bg-background">
      <Header 
        onSettingsClick={handleSettingsClick}
      />
      
      <main className="container mx-auto px-4 py-8">
        <IndexHeader />
        
        {showSettings ? (
          <div className="my-6">
            <WhatsAppSettings />
          </div>
        ) : (
          <TrackedStocksList 
            stocks={stocks}
            loading={loading}
            onToggleTracking={handleToggleTracking}
          />
        )}
      </main>
    </div>
  );
}
