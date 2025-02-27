
import { UserProfile, Session } from '@/utils/types';

export type UserContextType = {
  session: Session;
  signIn: (email: string, password: string) => Promise<{ error: any }>;
  signUp: (email: string, password: string, username: string) => Promise<{ error: any, requiresEmailConfirmation: boolean }>;
  signOut: () => Promise<void>;
  resendConfirmationEmail: (email: string) => Promise<{ error: any }>;
  resetPassword: (email: string) => Promise<{ error: any }>;
};
