
import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const IndexHeader: React.FC = () => {
  return <div className="flex justify-between items-center mb-6">
      <h1 className="text-2xl font-bold">Takip Ettiğiniz Hisseler</h1>
      
      <div className="flex space-x-2">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/stocks" className="flex items-center">
            Tüm Hisseler
            <ChevronRight className="h-4 w-4 ml-1" />
          </Link>
        </Button>
      </div>
    </div>;
};

export default IndexHeader;
