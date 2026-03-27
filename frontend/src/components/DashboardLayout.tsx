import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { MobileHeader } from './MobileHeader';
import { useChatStore } from '../stores/chatStore';

export function DashboardLayout() {
  const location = useLocation();
  const { isSidebarOpen, setSidebarOpen, isInsightsOpen, setInsightsOpen } = useChatStore();

  return (
    <div className="relative flex h-full w-full bg-background text-slate-100 overflow-hidden font-sans">
      {/* Background elements for depth */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-blue-500/10 rounded-full blur-[100px] pointer-events-none" />

      {/* Mobile Top Header */}
      <MobileHeader
        onToggleSidebar={() => {
          const nextState = !isSidebarOpen;
          setSidebarOpen(nextState);
          if (nextState) setInsightsOpen(false);
        }}
        onToggleInsights={() => {
          const nextState = !isInsightsOpen;
          setInsightsOpen(nextState);
          if (nextState) setSidebarOpen(false);
        }}
        isSidebarOpen={isSidebarOpen}
        isInsightsActive={isInsightsOpen}
        showInsightsAction={location.pathname === '/'}
      />

      {/* Main Layout Grid */}
      <div className="flex w-full h-full p-0 md:p-6 gap-0 md:gap-6 z-10 pt-[calc(4rem+env(safe-area-inset-top))] lg:pt-6">

        {/* Desktop Sidebar (Persistent) */}
        <Sidebar className="hidden lg:flex lg:w-[18%] lg:min-w-[240px]" />

        {/* Mobile Sidebar Drawer (Overlay) */}
        <AnimatePresence>
          {isSidebarOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setSidebarOpen(false)}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] lg:hidden"
              />
              {/* Drawer */}
              <motion.aside
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="fixed inset-y-0 left-0 w-[280px] bg-background z-[70] lg:hidden shadow-2xl border-r border-white/10 pt-16"
              >
                <div className="flex flex-col h-full">
                  <Sidebar
                    className="flex-1 !bg-transparent !border-0"
                    onItemClick={() => setSidebarOpen(false)}
                  />
                </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* Main Content Area */}
        <div className="flex-1 flex gap-0 md:gap-4 h-full min-w-0 overflow-hidden relative">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
