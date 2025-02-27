
import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { Session, UserProfile } from '@/utils/types';
import { toast } from 'sonner';

type UserContextType = {
  session: Session;
  signIn: (email: string, password: string) => Promise<{ error: any }>;
  signUp: (email: string, password: string, username: string) => Promise<{ error: any, requiresEmailConfirmation: boolean }>;
  signOut: () => Promise<void>;
  resendConfirmationEmail: (email: string) => Promise<{ error: any }>;
  resetPassword: (email: string) => Promise<{ error: any }>;
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
        // Check specifically for the email not confirmed error
        if (error.message.includes('Email not confirmed')) {
          toast.error('Email onaylanmamış. Lütfen onay e-postanızı kontrol edin veya yeni bir onay e-postası isteyin.');
        } else {
          toast.error(error.message);
        }
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
      // Get the current URL to use as a redirect base
      const baseUrl = window.location.origin;
      const redirectTo = `${baseUrl}/auth?mode=confirm`;

      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            username,
          },
          emailRedirectTo: redirectTo
        },
      });

      if (signUpError) {
        toast.error(signUpError.message);
        return { error: signUpError, requiresEmailConfirmation: false };
      }

      // Check if email confirmation is required
      if (data?.user?.identities && data.user.identities.length === 0) {
        // User already exists but hasn't confirmed their email
        toast.error('Bu email adresi zaten kayıtlı. Lütfen email adresinizi kontrol edin ve onaylayın.');
        return { error: null, requiresEmailConfirmation: true };
      }

      if (!data.user) {
        return { error: new Error('Kayıt işlemi başarısız'), requiresEmailConfirmation: false };
      }

      if (data.user && !data.session) {
        // Email confirmation is required
        toast.success('Kayıt başarılı! Lütfen email adresinizi kontrol edin ve hesabınızı onaylayın.');
        return { error: null, requiresEmailConfirmation: true };
      }

      // If we get here, user was created and automatically signed in (email confirmation disabled)
      toast.success('Kayıt başarılı, giriş yapıldı');
      return { error: null, requiresEmailConfirmation: false };
    } catch (error: any) {
      toast.error('Kayıt yapılamadı');
      return { error, requiresEmailConfirmation: false };
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

  // Updated function to resend confirmation email with correct redirect
  const resendConfirmationEmail = async (email: string) => {
    try {
      // Get the current URL to use as a redirect base
      const baseUrl = window.location.origin;
      const redirectTo = `${baseUrl}/auth?mode=confirm`;

      const { error } = await supabase.auth.resend({
        type: 'signup',
        email,
        options: {
          emailRedirectTo: redirectTo
        }
      });

      if (error) {
        console.error('Error resending confirmation email:', error);
        if (error.message.includes('For security purposes')) {
          toast.error('Güvenlik nedeniyle, kısa bir süre bekledikten sonra tekrar deneyin.');
        } else {
          toast.error(error.message);
        }
        return { error };
      }

      toast.success('Onay e-postası tekrar gönderildi. Lütfen e-posta kutunuzu kontrol edin.');
      return { error: null };
    } catch (error: any) {
      console.error('Error resending confirmation email:', error);
      toast.error('Onay e-postası gönderilemedi.');
      return { error };
    }
  };

  // Updated function to reset password with correct redirect
  const resetPassword = async (email: string) => {
    try {
      const baseUrl = window.location.origin;
      const redirectTo = `${baseUrl}/auth?mode=reset`;

      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: redirectTo,
      });

      if (error) {
        console.error('Error sending reset password email:', error);
        toast.error(error.message);
        return { error };
      }

      toast.success('Şifre sıfırlama e-postası gönderildi. Lütfen e-posta kutunuzu kontrol edin.');
      return { error: null };
    } catch (error: any) {
      console.error('Error sending reset password email:', error);
      toast.error('Şifre sıfırlama e-postası gönderilemedi.');
      return { error };
    }
  };

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
