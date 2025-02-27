
import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useUser } from '@/context/UserContext';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { MailIcon, AlertCircleIcon, ArrowLeftIcon } from 'lucide-react';

export default function Auth() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [showEmailConfirmation, setShowEmailConfirmation] = useState(false);
  const [showResetPassword, setShowResetPassword] = useState(false);
  const { session, signUp, resendConfirmationEmail, resetPassword } = useUser();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    // Redirect to home if already logged in
    if (session.user && !session.isLoading) {
      navigate('/');
    }

    // Check if we're in reset password mode from URL
    const mode = searchParams.get('mode');
    if (mode === 'reset') {
      setShowResetPassword(true);
    }
  }, [session, navigate, searchParams]);

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password || !username) return;
    
    setLoading(true);
    try {
      const { error, requiresEmailConfirmation } = await signUp(email, password, username);
      if (!error && requiresEmailConfirmation) {
        setShowEmailConfirmation(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResendConfirmation = async () => {
    if (!email) return;
    
    setLoading(true);
    try {
      await resendConfirmationEmail(email);
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    
    setLoading(true);
    try {
      await resetPassword(email);
    } finally {
      setLoading(false);
    }
  };

  if (session.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-pulse">Yükleniyor...</div>
      </div>
    );
  }

  if (showResetPassword) {
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
  
  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold">Haber Sinyalleri</h1>
          <p className="text-muted-foreground mt-2">Hisse haber takip uygulaması</p>
        </div>
        
        {showEmailConfirmation ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MailIcon className="h-5 w-5" />
                Email Onayı Gerekli
              </CardTitle>
              <CardDescription>
                Hesabınız oluşturuldu. Lütfen email adresinizi kontrol edin ve hesabınızı onaylayın.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <AlertDescription>
                  <span className="font-medium">{email}</span> adresine bir onay maili gönderdik. Lütfen mailinizi kontrol edin ve hesabınızı onaylayın.
                </AlertDescription>
              </Alert>
              
              <div className="rounded-lg border p-4">
                <h3 className="font-medium flex items-center gap-2">
                  <AlertCircleIcon className="h-4 w-4 text-amber-500" />
                  Onay e-postası almadınız mı?
                </h3>
                <p className="text-sm text-muted-foreground mt-1 mb-3">
                  Spam klasörünü kontrol edin veya yeni bir onay e-postası isteyin.
                </p>
                <Button 
                  variant="secondary" 
                  className="w-full" 
                  onClick={handleResendConfirmation}
                  disabled={loading}
                >
                  {loading ? "Gönderiliyor..." : "Yeni Onay E-postası Gönder"}
                </Button>
              </div>
            </CardContent>
            <CardFooter>
              <Button 
                variant="outline" 
                className="w-full" 
                onClick={() => setShowEmailConfirmation(false)}
              >
                Farklı bir email ile deneyin
              </Button>
            </CardFooter>
          </Card>
        ) : (
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
        )}
      </div>
    </div>
  );
}
