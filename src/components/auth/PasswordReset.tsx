
import React from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { ArrowLeftIcon } from 'lucide-react';

interface PasswordResetProps {
  email: string;
  setEmail: (email: string) => void;
  loading: boolean;
  handleResetPassword: (e: React.FormEvent) => Promise<void>;
  setShowResetPassword: (show: boolean) => void;
}

export function PasswordReset({ 
  email, 
  setEmail, 
  loading, 
  handleResetPassword, 
  setShowResetPassword 
}: PasswordResetProps) {
  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold">Haber Sinyalleri</h1>
          <p className="text-muted-foreground mt-2">Şifre Sıfırlama</p>
        </div>
        
        <Card>
          <CardHeader>
            <CardTitle>Şifre Sıfırlama</CardTitle>
            <CardDescription>
              E-posta adresinize bir şifre sıfırlama bağlantısı göndereceğiz.
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleResetPassword}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="reset-email">Email</Label>
                <Input
                  id="reset-email"
                  type="email"
                  placeholder="mail@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </CardContent>
            <CardFooter className="flex flex-col gap-4">
              <Button 
                type="submit" 
                className="w-full" 
                disabled={loading}
              >
                {loading ? "İşleniyor..." : "Şifre Sıfırlama Bağlantısı Gönder"}
              </Button>
              <Button 
                type="button" 
                variant="outline" 
                className="w-full" 
                onClick={() => setShowResetPassword(false)}
                disabled={loading}
              >
                <ArrowLeftIcon className="h-4 w-4 mr-2" />
                Kayıt Sayfasına Dön
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  );
}
