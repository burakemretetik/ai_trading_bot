
import React, { useState } from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { EmailSettings, SignalStrength } from '@/utils/types';
import { toast } from 'sonner';

interface EmailSetupProps {
  isOpen: boolean;
  onClose: () => void;
  settings: EmailSettings;
  onSave: (settings: EmailSettings) => void;
}

const EmailSetup: React.FC<EmailSetupProps> = ({
  isOpen,
  onClose,
  settings,
  onSave,
}) => {
  const [localSettings, setLocalSettings] = useState<EmailSettings>(settings);

  const handleSave = () => {
    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (localSettings.enabled && !emailRegex.test(localSettings.address)) {
      toast.error('Please enter a valid email address');
      return;
    }
    
    onSave(localSettings);
    toast.success('Email settings saved successfully');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in">
      <div className="w-full max-w-md bg-card rounded-lg shadow-lg border animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Email Notifications</h2>
          <Button variant="ghost" size="icon" className="rounded-full" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="email-notifications">Enable email notifications</Label>
              <p className="text-sm text-muted-foreground">
                Receive alerts when strong signals are detected
              </p>
            </div>
            <Switch
              id="email-notifications"
              checked={localSettings.enabled}
              onCheckedChange={(checked) =>
                setLocalSettings((prev) => ({ ...prev, enabled: checked }))
              }
            />
          </div>
          
          {localSettings.enabled && (
            <>
              <div className="space-y-2">
                <Label htmlFor="email">Email address</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="your@email.com"
                  value={localSettings.address}
                  onChange={(e) =>
                    setLocalSettings((prev) => ({
                      ...prev,
                      address: e.target.value,
                    }))
                  }
                />
              </div>
              
              <div className="space-y-3">
                <Label>Notification frequency</Label>
                <RadioGroup
                  value={localSettings.frequency}
                  onValueChange={(value: "instant" | "daily" | "weekly") =>
                    setLocalSettings((prev) => ({
                      ...prev,
                      frequency: value,
                    }))
                  }
                  className="flex flex-col gap-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="instant" id="instant" />
                    <Label htmlFor="instant">Instant (as signals are detected)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="daily" id="daily" />
                    <Label htmlFor="daily">Daily digest</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="weekly" id="weekly" />
                    <Label htmlFor="weekly">Weekly summary</Label>
                  </div>
                </RadioGroup>
              </div>
              
              <div className="space-y-3">
                <Label>Signal threshold</Label>
                <RadioGroup
                  value={localSettings.signalThreshold}
                  onValueChange={(value: SignalStrength) =>
                    setLocalSettings((prev) => ({
                      ...prev,
                      signalThreshold: value,
                    }))
                  }
                  className="flex flex-col gap-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="strong" id="strong" />
                    <Label htmlFor="strong">Strong signals only</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="medium" id="medium" />
                    <Label htmlFor="medium">Medium and strong signals</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="weak" id="weak" />
                    <Label htmlFor="weak">All signals (including weak)</Label>
                  </div>
                </RadioGroup>
              </div>
            </>
          )}
          
          <div className="flex justify-end pt-4">
            <Button variant="outline" className="mr-2" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave}>Save Settings</Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmailSetup;
