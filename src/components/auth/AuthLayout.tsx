
import React, { ReactNode } from 'react';

interface AuthLayoutProps {
  children: ReactNode;
  title?: string;
  subtitle?: string;
}

export function AuthLayout({ children, title = "Haber Sinyalleri", subtitle = "Hisse haber takip uygulaması" }: AuthLayoutProps) {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold">{title}</h1>
          <p className="text-muted-foreground mt-2">{subtitle}</p>
        </div>
        
        {children}
      </div>
    </div>
  );
}
