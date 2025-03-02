
import React, { createContext, useContext, useState, useEffect } from 'react';
import { UserContextType } from './types';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { formatPhoneNumber } from '@/services/userSettingsService';

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [phone, setPhone] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        setIsLoading(false);
        if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
          setIsAuthenticated(true);
          // Get phone from user metadata
          const phoneFromSession = session?.user?.phone || null;
          setPhone(phoneFromSession);
        } else if (event === 'SIGNED_OUT') {
          setIsAuthenticated(false);
          setPhone(null);
        }
      }
    );

    // Check current session on initial load
    const checkCurrentSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoading(false);
      if (session) {
        setIsAuthenticated(true);
        setPhone(session.user?.phone || null);
      }
    };

    checkCurrentSession();

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const signInWithPhone = async (phoneNumber: string) => {
    setIsLoading(true);
    const formattedPhone = formatPhoneNumber(phoneNumber);
    
    try {
      const { error } = await supabase.auth.signInWithOtp({
        phone: formattedPhone,
      });
      
      if (error) {
        toast.error(error.message);
        throw error;
      }
      
      toast.success('Verification code sent to your phone!');
    } catch (error) {
      console.error('Error during phone sign-in:', error);
      toast.error('Failed to send verification code');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyOTP = async (phoneNumber: string, token: string) => {
    setIsLoading(true);
    const formattedPhone = formatPhoneNumber(phoneNumber);
    
    try {
      const { error } = await supabase.auth.verifyOtp({
        phone: formattedPhone,
        token,
        type: 'sms',
      });
      
      if (error) {
        toast.error(error.message);
        throw error;
      }
      
      setPhone(formattedPhone);
      setIsAuthenticated(true);
      toast.success('Phone verified successfully!');
    } catch (error) {
      console.error('Error during OTP verification:', error);
      toast.error('Failed to verify code');
    } finally {
      setIsLoading(false);
    }
  };

  const signOut = async () => {
    setIsLoading(true);
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        toast.error(error.message);
        throw error;
      }
      setIsAuthenticated(false);
      setPhone(null);
      toast.success('Signed out successfully');
    } catch (error) {
      console.error('Error during sign out:', error);
      toast.error('Failed to sign out');
    } finally {
      setIsLoading(false);
    }
  };

  const contextValue: UserContextType = {
    isLoading,
    phone,
    isAuthenticated,
    signInWithPhone,
    verifyOTP,
    signOut
  };

  return (
    <UserContext.Provider value={contextValue}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};
