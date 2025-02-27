
import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Session, UserProfile } from '@/utils/types';
import { toast } from 'sonner';

type UserContextType = {
  session: Session;
  signIn: (email: string, password: string) => Promise<{ error: any }>;
  signUp: (email: string, password: string, username: string) => Promise<{ error: any }>;
  signOut: () => Promise<void>;
};

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider = ({ children }: { children: React.ReactNode }) => {
  const [session, setSession] = useState<Session>({
    user: null,
    isLoading: true,
  });

  useEffect(() => {
    // Check for active session on component mount
    const checkSession = async () => {
      try {
        const { data, error } = await supabase.auth.getSession();
        
        if (error) {
          console.error('Error fetching session:', error);
          return;
        }

        if (data.session) {
          const { data: profileData } = await supabase
            .from('profiles')
            .select('*')
            .eq('id', data.session.user.id)
            .single();

          setSession({
            user: profileData as UserProfile,
            isLoading: false,
          });
        } else {
          setSession({
            user: null,
            isLoading: false,
          });
        }
      } catch (error) {
        console.error('Session check error:', error);
        setSession({
          user: null,
          isLoading: false,
        });
      }
    };

    checkSession();

    // Subscribe to auth changes
    const { data: authListener } = supabase.auth.onAuthStateChange(async (event, session) => {
      console.log('Auth state changed:', event);
      
      if (session && event === 'SIGNED_IN') {
        const { data: profileData } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', session.user.id)
          .single();

        setSession({
          user: profileData as UserProfile,
          isLoading: false,
        });
      } else if (event === 'SIGNED_OUT') {
        setSession({
          user: null,
          isLoading: false,
        });
      }
    });

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, []);

  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      
      if (error) {
        toast.error(error.message);
        return { error };
      }
      
      toast.success('Giriş başarılı');
      return { error: null };
    } catch (error: any) {
      toast.error('Giriş yapılamadı');
      return { error };
    }
  };

  const signUp = async (email: string, password: string, username: string) => {
    try {
      // First sign up the user
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            username,
          },
        },
      });

      if (signUpError) {
        toast.error(signUpError.message);
        return { error: signUpError };
      }

      // If signup successful, immediately sign in
      if (data.user) {
        const { error: signInError } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (signInError) {
          toast.error('Kayıt başarılı fakat giriş yapılamadı');
          return { error: signInError };
        }
      }

      toast.success('Kayıt başarılı, giriş yapıldı');
      return { error: null };
    } catch (error: any) {
      toast.error('Kayıt yapılamadı');
      return { error };
    }
  };

  const signOut = async () => {
    try {
      await supabase.auth.signOut();
      toast.success('Çıkış yapıldı');
    } catch (error) {
      console.error('Sign out error:', error);
      toast.error('Çıkış yapılamadı');
    }
  };

  return (
    <UserContext.Provider
      value={{
        session,
        signIn,
        signUp,
        signOut,
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
