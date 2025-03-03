
import React, { useState, useEffect } from 'react';
import { Stock } from '@/utils/types';
import { mockStocks, createMockStocksFromCSV } from '@/utils/mockData';
import { toast } from 'sonner';
import { getTrackedStocks, trackStock, untrackStock } from '@/services/stockService';
import StockListHeader from '@/components/stock/StockListHeader';
import FollowAllToggle from '@/components/stock/FollowAllToggle';
import StockSearchBar from '@/components/stock/StockSearchBar';
import StockListContent from '@/components/stock/StockListContent';
import LoadingSkeleton from '@/components/stock/LoadingSkeleton';

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

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  return (
    <div className="min-h-screen bg-background">
      <StockListHeader />
      
      <main className="max-w-7xl mx-auto px-4 py-8">
        {loading ? (
          <LoadingSkeleton />
        ) : (
          <div className="space-y-4">
            <div className="flex justify-between mb-6">
              <h2 className="text-lg font-medium">Takip edebileceğiniz hisseler</h2>
              <div className="text-sm text-muted-foreground">
                {stocks.filter(s => s.tracked).length} takip edilen hisse
              </div>
            </div>
            
            <FollowAllToggle 
              isFollowAllActive={isFollowAllActive} 
              onToggleFollowAll={handleToggleFollowAll} 
            />
            
            <StockSearchBar 
              searchQuery={searchQuery} 
              onSearchChange={handleSearchChange} 
            />
            
            <StockListContent 
              filteredStocks={filteredStocks} 
              searchQuery={searchQuery} 
              onToggleTracking={handleToggleTracking} 
            />
          </div>
        )}
      </main>
    </div>
  );
};

export default StockList;
