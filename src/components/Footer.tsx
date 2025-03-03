
import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="border-t mt-auto py-4 bg-background">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Stock News Signal
          </div>
          <div className="flex space-x-4 mt-2 md:mt-0">
            <Link 
              to="/privacy" 
              className="text-sm text-muted-foreground hover:text-primary hover:underline"
            >
              Gizlilik Politikası
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
