
import React from 'react';
import { Link } from 'react-router-dom';
import { BellRing, Search, List, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  onSearchClick: () => void;
  onSettingsClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ onSearchClick, onSettingsClick }) => {
  return (
    <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <BellRing className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-semibold">Hisse Haberleri</h1>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onSearchClick}
            aria-label="Hisse Ara"
          >
            <Search className="h-5 w-5" />
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            asChild
            aria-label="Hisse Listesi"
          >
            <Link to="/stocks">
              <List className="h-5 w-5" />
            </Link>
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={onSettingsClick}
            aria-label="Ayarlar"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
