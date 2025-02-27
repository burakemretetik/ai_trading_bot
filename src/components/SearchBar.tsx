
import React, { useState, useRef, useEffect } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Stock } from '@/utils/types';
import { useQuery } from '@tanstack/react-query';
import { stockApi } from '@/services/stockApi';

interface SearchBarProps {
  isOpen: boolean;
  onClose: () => void;
  onAddStock: (stock: Stock) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ isOpen, onClose, onAddStock }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Stock[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch all stocks
  const { data: stocks, isLoading, isError } = useQuery({
    queryKey: ['search-stocks'],
    queryFn: async () => {
      const response = await stockApi.getAllStocks();
      if (!response.success) {
        throw new Error(response.error);
      }
      return response.data;
    },
    // Only fetch when search modal is open
    enabled: isOpen,
  });

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (stocks && query.length > 0) {
      const filtered = stocks.filter(
        stock => 
          stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
          stock.name.toLowerCase().includes(query.toLowerCase())
      );
      setResults(filtered);
    } else {
      setResults([]);
    }
  }, [query, stocks]);

  const handleAddStock = (stock: Stock) => {
    onAddStock(stock);
    setQuery('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-start justify-center pt-20 animate-fade-in">
      <div className="w-full max-w-md bg-card rounded-lg shadow-lg border animate-slide-up">
        <div className="flex items-center p-4 border-b">
          <Search className="h-5 w-5 text-muted-foreground mr-2" />
          <Input
            ref={inputRef}
            type="text"
            placeholder="Şirket adı veya sembol arayın..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
          />
          <Button 
            variant="ghost" 
            size="icon" 
            className="rounded-full ml-2" 
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="py-8 flex justify-center items-center">
              <Loader2 className="h-8 w-8 text-primary animate-spin" />
            </div>
          ) : isError ? (
            <div className="py-8 text-center">
              <p className="text-red-500">Hisse verileri yüklenemedi.</p>
            </div>
          ) : results.length > 0 ? (
            <ul className="py-2">
              {results.map(stock => (
                <li key={stock.id} className="px-4 py-3 hover:bg-secondary/50 transition-colors cursor-pointer">
                  <div className="flex items-center justify-between" onClick={() => handleAddStock(stock)}>
                    <div>
                      <span className="font-semibold">{stock.symbol}</span>
                      <p className="text-sm text-muted-foreground">{stock.name}</p>
                    </div>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className={stock.tracked ? "text-muted-foreground" : ""}
                    >
                      {stock.tracked ? 'Takipte' : 'Ekle'}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          ) : query.length > 0 ? (
            <div className="py-8 text-center">
              <p className="text-muted-foreground">"{query}" için sonuç bulunamadı</p>
            </div>
          ) : (
            <div className="py-8 text-center">
              <p className="text-muted-foreground">Hisse aramak için yazmaya başlayın</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchBar;
