import { createBrowserRouter, Outlet, RouterProvider } from "react-router-dom";
import { Navbar } from "./components/layout/Navbar";
import { AdminPage } from "./pages/AdminPage";
import { DashboardPage } from "./pages/DashboardPage";
import { SearchPage } from "./pages/SearchPage";

function RootLayout() {
  return (
    <div className="min-h-screen bg-surface text-text">
      <Navbar />
      <Outlet />
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <RootLayout />,
    children: [
      { path: "/", element: <SearchPage /> },
      { path: "/dashboard", element: <DashboardPage /> },
      { path: "/admin", element: <AdminPage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
