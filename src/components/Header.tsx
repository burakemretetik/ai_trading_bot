
import React from 'react';
import { Search, Settings, User, LogOut } from 'lucide-react';
import { Button } from './ui/button';
import { useUser } from '@/context/UserContext';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

type HeaderProps = {
  onSearchClick: () => void;
  onSettingsClick: () => void;
};

const Header = ({ onSearchClick, onSettingsClick }: HeaderProps) => {
  const { session, signOut } = useUser();

  const handleSignOut = async () => {
    await signOut();
  };

  return (
    <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-10">
      <div className="flex h-16 items-center px-4 container mx-auto">
        <div className="text-xl font-semibold text-foreground">
          Haber Sinyalleri
        </div>
        <div className="ml-auto flex items-center space-x-2">
          <Button variant="ghost" size="icon" onClick={onSearchClick}>
            <Search className="h-5 w-5" />
            <span className="sr-only">Hisse Ara</span>
          </Button>
          <Button variant="ghost" size="icon" onClick={onSettingsClick}>
            <Settings className="h-5 w-5" />
            <span className="sr-only">Ayarlar</span>
          </Button>
          
          {session.user && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon">
                  <User className="h-5 w-5" />
                  <span className="sr-only">Kullanıcı Menüsü</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>
                  {session.user.username || 'Kullanıcı'}
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Çıkış Yap
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
