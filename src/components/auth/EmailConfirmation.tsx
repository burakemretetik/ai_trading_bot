
import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { MailIcon, AlertCircleIcon } from 'lucide-react';

interface EmailConfirmationProps {
  email: string;
  loading: boolean;
  handleResendConfirmation: () => Promise<void>;
  setShowEmailConfirmation: (show: boolean) => void;
}

export function EmailConfirmation({ 
  email, 
  loading, 
  handleResendConfirmation, 
  setShowEmailConfirmation 
}: EmailConfirmationProps) {
  return (
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
  );
}
