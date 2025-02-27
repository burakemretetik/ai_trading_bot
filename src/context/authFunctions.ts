
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { Session, UserProfile } from '@/utils/types';

// Function to sign in a user
export const signIn = async (email: string, password: string) => {
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

// Function to sign up a user
export const signUp = async (email: string, password: string, username: string) => {
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

// Function to sign out a user
export const signOut = async () => {
  try {
    await supabase.auth.signOut();
    toast.success('Çıkış yapıldı');
  } catch (error) {
    console.error('Sign out error:', error);
    toast.error('Çıkış yapılamadı');
  }
};

// Function to resend confirmation email
export const resendConfirmationEmail = async (email: string) => {
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

// Function to reset password
export const resetPassword = async (email: string) => {
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

// Function to fetch the user profile
export const fetchUserProfile = async (userId: string): Promise<UserProfile | null> => {
  try {
    const { data: profileData, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single();
    
    if (error) {
      console.error('Error fetching user profile:', error);
      return null;
    }
    
    return profileData as UserProfile;
  } catch (error) {
    console.error('Error fetching user profile:', error);
    return null;
  }
};
