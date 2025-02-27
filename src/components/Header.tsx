
import React from 'react';
import { Bell, Settings, Search } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  onSearchClick: () => void;
  onSettingsClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onSearchClick, onSettingsClick }) => {
  return (
    <header className="sticky top-0 z-10 w-full glass-effect border-b border-gray-200 px-4 py-3 animate-fade-in">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-semibold text-sm">S</span>
          </div>
          <h1 className="text-xl font-semibold tracking-tight">Stock News Signal</h1>
        </div>
        
        <div className="flex items-center space-x-3">
          <Button 
            variant="outline" 
            size="sm" 
            className="rounded-full transition-all duration-300 hover:bg-secondary"
            onClick={onSearchClick}
          >
            <Search className="h-4 w-4 mr-2" />
            <span>Search</span>
          </Button>
          
          <Button 
            variant="ghost" 
            size="icon" 
            className="rounded-full transition-all duration-300"
            onClick={onSettingsClick}
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
