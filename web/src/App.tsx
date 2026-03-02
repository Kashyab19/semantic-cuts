import { createBrowserRouter, Outlet, RouterProvider } from "react-router-dom";
import { Navbar } from "./components/layout/Navbar";
import { useTheme } from "./hooks/useTheme";
import { AdminPage } from "./pages/AdminPage";
import { DashboardPage } from "./pages/DashboardPage";
import { LandingPage } from "./pages/LandingPage";
import { SearchPage } from "./pages/SearchPage";

function LandingLayout() {
  return (
    <div className="min-h-screen bg-surface text-text">
      <Outlet />
    </div>
  );
}

function AppLayout() {
  return (
    <div className="min-h-screen bg-surface text-text">
      <Navbar />
      <Outlet />
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <LandingLayout />,
    children: [{ path: "/", element: <LandingPage /> }],
  },
  {
    element: <AppLayout />,
    children: [
      { path: "/playground", element: <SearchPage /> },
      { path: "/dashboard", element: <DashboardPage /> },
      { path: "/admin", element: <AdminPage /> },
    ],
  },
]);

export default function App() {
  useTheme();
  return <RouterProvider router={router} />;
}
