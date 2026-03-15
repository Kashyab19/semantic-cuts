import { createBrowserRouter, Outlet, RouterProvider } from "react-router-dom";
import { Navbar } from "./components/layout/Navbar";
import { useTheme } from "./hooks/useTheme";
import { LandingPage } from "./pages/LandingPage";
import { SearchPage } from "./pages/SearchPage";
import { ProcessPage } from "./pages/ProcessPage";
import { GalleryPage } from "./pages/GalleryPage";
import { StatsPage } from "./pages/StatsPage";

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
      { path: "/search", element: <SearchPage /> },
      { path: "/process", element: <ProcessPage /> },
      { path: "/gallery", element: <GalleryPage /> },
      { path: "/stats", element: <StatsPage /> },
    ],
  },
]);

export default function App() {
  useTheme();
  return <RouterProvider router={router} />;
}
