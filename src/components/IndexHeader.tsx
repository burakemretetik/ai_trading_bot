
import React from 'react';
import { Link } from 'react-router-dom';
import { Plus, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface IndexHeaderProps {
  onAddStock: () => void;
}

const IndexHeader: React.FC<IndexHeaderProps> = ({ onAddStock }) => {
  return (
    <div className="flex justify-between items-center mb-6">
      <h1 className="text-2xl font-bold">Takip Ettiğiniz Hisselerin Güncel Haberleri</h1>
      
      <div className="flex space-x-2">
        <Button variant="outline" size="sm" onClick={onAddStock}>
          <Plus className="h-4 w-4 mr-2" />
          Hisse Ekle
        </Button>
        
        <Button variant="ghost" size="sm" asChild>
          <Link to="/stocks" className="flex items-center">
            Tüm Hisseler
            <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default IndexHeader;
