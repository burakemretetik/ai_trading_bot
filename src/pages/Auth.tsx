
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '@/context/UserContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { AuthLayout } from '@/components/auth/AuthLayout';
import { formatPhoneNumber } from '@/services/userSettingsService';

export default function Auth() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading, signInWithPhone, verifyOTP } = useUser();
  const [phoneNumber, setPhoneNumber] = useState('');
  const [verificationCode, setVerificationCode] = useState('');
  const [codeSent, setCodeSent] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      navigate('/');
    }
  }, [isAuthenticated, isLoading, navigate]);

  const handleSendCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!phoneNumber.trim()) return;
    
    setSubmitting(true);
    try {
      await signInWithPhone(phoneNumber);
      setCodeSent(true);
    } finally {
      setSubmitting(false);
    }
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!verificationCode.trim()) return;
    
    setSubmitting(true);
    try {
      await verifyOTP(phoneNumber, verificationCode);
      // Navigation to home is handled by the useEffect
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout>
      <Card>
        <CardHeader>
          <CardTitle>Phone Authentication</CardTitle>
          <CardDescription>
            {codeSent 
              ? 'Enter the verification code sent to your phone'
              : 'Sign in or sign up with your phone number'}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!codeSent ? (
            <form onSubmit={handleSendCode} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number</Label>
                <Input
                  id="phone"
                  type="tel"
                  placeholder="+90 5XX XXX XX XX"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Enter your phone number with country code (e.g., +90 for Turkey)
                </p>
              </div>
              <Button 
                type="submit" 
                className="w-full" 
                disabled={submitting || isLoading}
              >
                {submitting ? 'Sending...' : 'Send Verification Code'}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleVerifyCode} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="verificationCode">Verification Code</Label>
                <Input
                  id="verificationCode"
                  type="text"
                  placeholder="Enter 6-digit code"
                  value={verificationCode}
                  onChange={(e) => setVerificationCode(e.target.value)}
                  maxLength={6}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Enter the 6-digit verification code sent to {formatPhoneNumber(phoneNumber)}
                </p>
              </div>
              <Button 
                type="submit" 
                className="w-full" 
                disabled={submitting || isLoading}
              >
                {submitting ? 'Verifying...' : 'Verify Code'}
              </Button>
              <Button 
                type="button" 
                variant="ghost" 
                className="w-full" 
                onClick={() => setCodeSent(false)}
              >
                Change Phone Number
              </Button>
            </form>
          )}
        </CardContent>
      </Card>
    </AuthLayout>
  );
}
