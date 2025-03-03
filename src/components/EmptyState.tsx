
import React from 'react';
import { List, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface EmptyStateProps {
  onSearchClick: () => void;
}

const EmptyState: React.FC<EmptyStateProps> = ({ onSearchClick }) => {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] max-w-md mx-auto text-center p-4 animate-fade-in">
      <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mb-6">
        <TrendingUp className="h-8 w-8 text-primary" />
      </div>
      
      <h2 className="text-2xl font-semibold mb-2">Henüz takip edilen hisse yok</h2>
      <p className="text-muted-foreground mb-6">
        Haberler ve piyasa bilgilerini almak için hisseleri takip listenize ekleyin.
      </p>
      
      <Button onClick={onSearchClick} size="lg" className="gap-2">
        <List className="h-4 w-4" />
        <span>Tüm Hisseler</span>
      </Button>
    </div>
  );
};

export default EmptyState;
