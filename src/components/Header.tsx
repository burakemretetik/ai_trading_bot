
import React from 'react';
import { Settings } from 'lucide-react';
import { Button } from './ui/button';

type HeaderProps = {
  onSettingsClick: () => void;
};

const Header = ({ onSettingsClick }: HeaderProps) => {
  return (
    <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
      <div className="flex h-16 items-center px-4 container mx-auto">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          Haber Sinyalleri
        </h1>
        <div className="ml-auto flex items-center space-x-2">
          <Button variant="ghost" size="icon" onClick={onSettingsClick}>
            <Settings className="h-5 w-5" />
            <span className="sr-only">Ayarlar</span>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;
