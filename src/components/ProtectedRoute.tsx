
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '@/context/UserContext';

type ProtectedRouteProps = {
  children: React.ReactNode;
};

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const { session } = useUser();
  const navigate = useNavigate();

  useEffect(() => {
    if (!session.isLoading && !session.user) {
      navigate('/auth');
    }
  }, [session, navigate]);

  if (session.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-pulse">Yükleniyor...</div>
      </div>
    );
  }

  return session.user ? <>{children}</> : null;
};

export default ProtectedRoute;
