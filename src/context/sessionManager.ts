
import { supabase } from '@/integrations/supabase/client';
import { Session, UserProfile } from '@/utils/types';
import { fetchUserProfile } from './authFunctions';

// Function to check the current session
export const checkSession = async (): Promise<Session> => {
  try {
    const { data, error } = await supabase.auth.getSession();
    
    if (error) {
      console.error('Error fetching session:', error);
      return { user: null, isLoading: false };
    }

    if (data.session) {
      const profileData = await fetchUserProfile(data.session.user.id);
      
      return {
        user: profileData,
        isLoading: false,
      };
    }
    
    return {
      user: null,
      isLoading: false,
    };
  } catch (error) {
    console.error('Session check error:', error);
    return {
      user: null,
      isLoading: false,
    };
  }
};

// Function to set up auth state change listener
export const setupAuthListener = (
  setSession: (session: Session) => void
) => {
  const { data: authListener } = supabase.auth.onAuthStateChange(async (event, session) => {
    console.log('Auth state changed:', event);
    
    if (session && event === 'SIGNED_IN') {
      const profileData = await fetchUserProfile(session.user.id);
      
      setSession({
        user: profileData,
        isLoading: false,
      });
    } else if (event === 'SIGNED_OUT') {
      setSession({
        user: null,
        isLoading: false,
      });
    }
  });

  return authListener;
};
