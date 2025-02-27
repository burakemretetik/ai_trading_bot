
import React from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';

interface SignUpFormProps {
  email: string;
  setEmail: (email: string) => void;
  password: string;
  setPassword: (password: string) => void;
  username: string;
  setUsername: (username: string) => void;
  loading: boolean;
  handleSignUp: (e: React.FormEvent) => Promise<void>;
  setShowResetPassword: (show: boolean) => void;
}

export function SignUpForm({ 
  email, 
  setEmail, 
  password, 
  setPassword, 
  username, 
  setUsername, 
  loading, 
  handleSignUp,
  setShowResetPassword
}: SignUpFormProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Kayıt Ol</CardTitle>
        <CardDescription>
          Hisse takibi yapmak için yeni bir hesap oluşturun.
        </CardDescription>
      </CardHeader>
      <form onSubmit={handleSignUp}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="signup-email">Email</Label>
            <Input
              id="signup-email"
              type="email"
              placeholder="mail@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="username">Kullanıcı Adı</Label>
            <Input
              id="username"
              type="text"
              placeholder="kullanici_adi"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="signup-password">Şifre</Label>
            <Input
              id="signup-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div className="text-sm text-right">
            <Button 
              type="button" 
              variant="link" 
              className="p-0 h-auto" 
              onClick={() => setShowResetPassword(true)}
            >
              Şifremi unuttum
            </Button>
          </div>
        </CardContent>
        <CardFooter>
          <Button 
            type="submit" 
            className="w-full" 
            disabled={loading}
          >
            {loading ? "Kayıt yapılıyor..." : "Kayıt Ol"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}
