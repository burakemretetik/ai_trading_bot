
export type UserContextType = {
  isLoading: boolean;
  phone: string | null;
  isAuthenticated: boolean;
  signInWithPhone: (phone: string) => Promise<void>;
  verifyOTP: (phone: string, token: string) => Promise<void>;
  signOut: () => Promise<void>;
};
