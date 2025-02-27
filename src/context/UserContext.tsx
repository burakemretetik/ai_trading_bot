
import React, { createContext, useContext, useEffect, useState } from 'react';
import { Session } from '@/utils/types';
import { UserContextType } from './types';
import { 
  signIn, 
  signUp, 
  signOut, 
  resendConfirmationEmail, 
  resetPassword 
} from './authFunctions';
import { checkSession, setupAuthListener } from './sessionManager';

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [session, setSession] = useState<Session>({
    user: null,
    isLoading: true,
  });

  useEffect(() => {
    // Check for active session on component mount
    const initializeSession = async () => {
      const sessionData = await checkSession();
      setSession(sessionData);
    };

    initializeSession();

    // Subscribe to auth changes
    const authListener = setupAuthListener(setSession);

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, []);

  return (
    <UserContext.Provider
      value={{
        session,
        signIn,
        signUp,
        signOut,
        resendConfirmationEmail,
        resetPassword,
      }}
    >
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
