
import React from 'react';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';

interface StockSearchBarProps {
  searchQuery: string;
  onSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const StockSearchBar: React.FC<StockSearchBarProps> = ({ 
  searchQuery, 
  onSearchChange 
}) => {
  return (
    <div className="relative mb-6">
      <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
        <Search className="h-4 w-4 text-muted-foreground" />
      </div>
      <Input 
        type="text" 
        placeholder="Hisse adı veya sembol ara..." 
        value={searchQuery} 
        onChange={onSearchChange} 
        className="pl-10" 
      />
    </div>
  );
};

export default StockSearchBar;
