import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function DashboardLayout() {
  return (
    <div className="relative flex h-full w-full bg-background text-slate-100 overflow-hidden font-sans">
      {/* Background elements for depth */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-blue-500/10 rounded-full blur-[100px] pointer-events-none" />

      {/* Main Layout Grid */}
      <div className="flex w-full h-full p-4 gap-4 z-10">
        <Sidebar className="w-[18%] min-w-[240px]" />

        {/* Render child routes here */}
        <div className="flex-1 flex gap-4 h-full min-w-0 overflow-hidden">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
