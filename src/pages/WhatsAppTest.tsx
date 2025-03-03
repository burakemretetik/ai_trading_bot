
import React, { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import WhatsAppSettings from '@/components/WhatsAppSettings';
import Header from '@/components/Header';
import { getUserSettings, hasValidPhoneNumber } from '@/services/userSettingsService';
import { checkForNewsAndNotifyUser } from '@/services/newsService';

const WhatsAppTest = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [hasPhone, setHasPhone] = useState(false);

  useEffect(() => {
    setHasPhone(hasValidPhoneNumber());
  }, []);

  const handleTestNotification = async () => {
    if (!hasValidPhoneNumber()) {
      toast.warning('Lütfen önce telefon numaranızı ayarlayın', {
        duration: 4000,
      });
      return;
    }

    setIsLoading(true);
    try {
      // Pass forceNotify as true to force a test notification even if there's no news
      const result = await checkForNewsAndNotifyUser(true);
      if (result) {
        toast.success('WhatsApp bildirimi gönderildi', {
          duration: 4000,
        });
      } else {
        toast.info('Bildirim gönderilirken bir sorun oluştu', {
          duration: 4000,
        });
      }
    } catch (error) {
      console.error('Error testing WhatsApp notification:', error);
      toast.error('WhatsApp bildirimi gönderilirken bir hata oluştu', {
        duration: 4000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4">
      <Header onSettingsClick={() => {}} />
      
      <div className="my-8 space-y-6">
        <h1 className="text-2xl font-bold">WhatsApp Bildirim Testi</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>WhatsApp Ayarları</CardTitle>
            </CardHeader>
            <CardContent>
              <WhatsAppSettings />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Test Bildirimi</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                Bu buton, telefon numaranıza test amaçlı bir WhatsApp bildirimi gönderir.
              </p>
              
              {!hasPhone && (
                <div className="bg-yellow-50 border border-yellow-300 text-yellow-800 p-3 rounded-md mb-4">
                  <p className="text-sm">Bildirim testi yapabilmek için önce telefonunuzu ekleyin ve WhatsApp bildirimlerini etkinleştirin.</p>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <Button 
                onClick={handleTestNotification} 
                disabled={isLoading || !hasPhone}
                className="w-full"
              >
                {isLoading ? 'Gönderiliyor...' : 'WhatsApp Test Bildirimi Gönder'}
              </Button>
            </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default WhatsAppTest;
