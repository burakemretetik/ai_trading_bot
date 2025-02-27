
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { AuthLayout } from '@/components/auth/AuthLayout';

export default function Auth() {
  const navigate = useNavigate();

  const handleContinue = () => {
    navigate('/');
  };

  return (
    <AuthLayout>
      <Card>
        <CardHeader>
          <CardTitle>Welcome</CardTitle>
          <CardDescription>
            Authentication has been removed from this application.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p>The application now works without requiring any user authentication.</p>
        </CardContent>
        <CardFooter>
          <Button 
            className="w-full" 
            onClick={handleContinue}
          >
            Continue to Application
          </Button>
        </CardFooter>
      </Card>
    </AuthLayout>
  );
}
