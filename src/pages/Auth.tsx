
import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useUser } from '@/context/UserContext';
import { AuthLayout } from '@/components/auth/AuthLayout';
import { SignUpForm } from '@/components/auth/SignUpForm';
import { EmailConfirmation } from '@/components/auth/EmailConfirmation';
import { PasswordReset } from '@/components/auth/PasswordReset';

export default function Auth() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [showEmailConfirmation, setShowEmailConfirmation] = useState(false);
  const [showResetPassword, setShowResetPassword] = useState(false);
  const { session, signUp, resendConfirmationEmail, resetPassword } = useUser();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    // Redirect to home if already logged in
    if (session.user && !session.isLoading) {
      navigate('/');
    }

    // Check if we're in reset password mode from URL
    const mode = searchParams.get('mode');
    if (mode === 'reset') {
      setShowResetPassword(true);
    }
  }, [session, navigate, searchParams]);

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password || !username) return;
    
    setLoading(true);
    try {
      const { error, requiresEmailConfirmation } = await signUp(email, password, username);
      if (!error && requiresEmailConfirmation) {
        setShowEmailConfirmation(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResendConfirmation = async () => {
    if (!email) return;
    
    setLoading(true);
    try {
      await resendConfirmationEmail(email);
    } finally {
      setLoading(false);
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) return;
    
    setLoading(true);
    try {
      await resetPassword(email);
    } finally {
      setLoading(false);
    }
  };

  if (session.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-pulse">Yükleniyor...</div>
      </div>
    );
  }

  if (showResetPassword) {
    return (
      <PasswordReset 
        email={email}
        setEmail={setEmail}
        loading={loading}
        handleResetPassword={handleResetPassword}
        setShowResetPassword={setShowResetPassword}
      />
    );
  }
  
  return (
    <AuthLayout>
      {showEmailConfirmation ? (
        <EmailConfirmation 
          email={email}
          loading={loading}
          handleResendConfirmation={handleResendConfirmation}
          setShowEmailConfirmation={setShowEmailConfirmation}
        />
      ) : (
        <SignUpForm 
          email={email}
          setEmail={setEmail}
          password={password}
          setPassword={setPassword}
          username={username}
          setUsername={setUsername}
          loading={loading}
          handleSignUp={handleSignUp}
          setShowResetPassword={setShowResetPassword}
        />
      )}
    </AuthLayout>
  );
}
