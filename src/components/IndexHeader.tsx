
import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const IndexHeader: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-xl font-semibold tracking-tight">Takip Ettiğiniz Hisseler</h1>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/stocks" className="flex items-center">
              Tüm Hisseler
              <ChevronRight className="h-4 w-4 ml-1" />
            </Link>
          </Button>
        </div>
      </div>
      <h2 className="text-sm text-muted-foreground">Favorilerinize eklediğiniz hisseler</h2>
    </div>
  );
};

export default IndexHeader;
