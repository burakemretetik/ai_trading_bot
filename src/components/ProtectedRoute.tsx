
type ProtectedRouteProps = {
  children: React.ReactNode;
};

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  // Since we've removed authentication, we simply render the children
  return <>{children}</>;
};

export default ProtectedRoute;
