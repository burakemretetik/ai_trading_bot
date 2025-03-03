
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';
import { Link } from 'react-router-dom';
import { 
  getUserSettings, 
  saveUserSettings, 
  formatPhoneNumber
} from '@/services/userSettingsService';
import { UserSettings } from '@/utils/types';

const WhatsAppSettings = () => {
  const [settings, setSettings] = useState<UserSettings>({
    phoneNumber: '',
    whatsappEnabled: true
  });
  
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    // Load user settings when component mounts
    const userSettings = getUserSettings();
    setSettings(userSettings);
  }, []);
  
  const handlePhoneNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSettings(prev => ({
      ...prev,
      phoneNumber: e.target.value
    }));
  };
  
  const handleToggleWhatsApp = (checked: boolean) => {
    setSettings(prev => ({
      ...prev,
      whatsappEnabled: checked
    }));
  };
  
  const handleSaveSettings = () => {
    setIsLoading(true);
    
    try {
      // Format phone number before saving
      const formattedPhoneNumber = formatPhoneNumber(settings.phoneNumber);
      
      const success = saveUserSettings({
        ...settings,
        phoneNumber: formattedPhoneNumber
      });
      
      if (success) {
        toast.success('WhatsApp ayarları kaydedildi');
      } else {
        toast.error('Ayarlar kaydedilirken bir hata oluştu');
      }
    } catch (error) {
      console.error('Error saving settings:', error);
      toast.error('Bir hata oluştu');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="space-y-6 p-4 bg-card rounded-lg border shadow-sm">
      <div className="space-y-2">
        <h3 className="text-lg font-medium">WhatsApp Bildirimleri</h3>
        <p className="text-sm text-muted-foreground">
          Takip ettiğiniz hisseler hakkında yeni haberler WhatsApp üzerinden size bildirilecektir.
        </p>
      </div>
      
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="whatsapp-enabled" className="flex flex-col space-y-1">
            <span>WhatsApp Bildirimleri</span>
            <span className="font-normal text-xs text-muted-foreground">
              Hisse haberleri için WhatsApp bildirimleri alın
            </span>
          </Label>
          <Switch 
            id="whatsapp-enabled"
            checked={settings.whatsappEnabled}
            onCheckedChange={handleToggleWhatsApp}
          />
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="phone-number">Telefon Numarası</Label>
          <Input
            id="phone-number"
            placeholder="+90 5XX XXX XX XX"
            value={settings.phoneNumber}
            onChange={handlePhoneNumberChange}
            disabled={!settings.whatsappEnabled}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Lütfen WhatsApp'ta kullandığınız telefon numaranızı uluslararası formatta girin (+90...)
          </p>
        </div>
        
        <Button 
          onClick={handleSaveSettings} 
          disabled={isLoading || !settings.whatsappEnabled}
          className="w-full"
        >
          {isLoading ? 'Kaydediliyor...' : 'Kaydet'}
        </Button>
        
        <div className="pt-2 text-center">
          <Link to="/privacy" className="text-xs text-muted-foreground hover:text-primary hover:underline">
            Gizlilik Politikası
          </Link>
        </div>
      </div>
    </div>
  );
};

export default WhatsAppSettings;
