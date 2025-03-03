
import React from 'react';
import { CheckSquare } from 'lucide-react';
import { Switch } from '@/components/ui/switch';

interface FollowAllToggleProps {
  isFollowAllActive: boolean;
  onToggleFollowAll: () => void;
}

const FollowAllToggle: React.FC<FollowAllToggleProps> = ({ 
  isFollowAllActive, 
  onToggleFollowAll 
}) => {
  return (
    <div className="flex items-center justify-between p-3 bg-card rounded-lg border mb-4">
      <div className="flex items-center gap-2">
        <CheckSquare className="h-5 w-5 text-primary" />
        <span className="font-medium">Tüm hisseleri takip et</span>
      </div>
      <Switch 
        checked={isFollowAllActive} 
        onCheckedChange={onToggleFollowAll}
        aria-label="Tüm hisseleri takip et"
      />
    </div>
  );
};

export default FollowAllToggle;
