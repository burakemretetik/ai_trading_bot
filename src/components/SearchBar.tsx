
import React, { useState, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { mockStocks } from '@/utils/mockData';
import { Stock } from '@/utils/types';

interface SearchBarProps {
  isOpen: boolean;
  onClose: () => void;
  onAddStock: (stock: Stock) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ isOpen, onClose, onAddStock }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Stock[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (query.length > 0) {
      const filtered = mockStocks.filter(
        stock => 
          stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
          stock.name.toLowerCase().includes(query.toLowerCase())
      );
      setResults(filtered);
    } else {
      setResults([]);
    }
  }, [query]);

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
            placeholder="Search for companies or symbols..."
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
          {results.length > 0 ? (
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
                      {stock.tracked ? 'Tracked' : 'Add'}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          ) : query.length > 0 ? (
            <div className="py-8 text-center">
              <p className="text-muted-foreground">No results found for "{query}"</p>
            </div>
          ) : (
            <div className="py-8 text-center">
              <p className="text-muted-foreground">Start typing to search for stocks</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchBar;
