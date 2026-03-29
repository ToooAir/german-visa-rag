import { useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { MobileHeader } from './MobileHeader';
import { useChatStore } from '../stores/chatStore';

export function DashboardLayout() {
  const location = useLocation();
  const { isSidebarOpen, setSidebarOpen, isInsightsOpen, setInsightsOpen } = useChatStore();

  // 路由切換時，強制關閉面板防鎖死
  useEffect(() => {
    setSidebarOpen(false);
    setInsightsOpen(false);
  }, [location.pathname, setSidebarOpen, setInsightsOpen]);

  // Scroll lock — 不動 body.style.position
  useEffect(() => {
    const shouldLock = (isSidebarOpen || isInsightsOpen) && window.innerWidth < 1024;
    const root = document.getElementById('root');
    if (root) root.style.overflow = shouldLock ? 'hidden' : '';
    return () => { if (root) root.style.overflow = ''; };
  }, [isSidebarOpen, isInsightsOpen]);

  // theme-color 動態同步 sidebar 背景色
  useEffect(() => {
    const meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) return;
    const isDark = document.documentElement.classList.contains('dark');
    if (isSidebarOpen || isInsightsOpen) {
      // 動態對應 Drawer 實體色
      meta.setAttribute('content', isDark ? '#0B1120' : '#f8fafc');
    } else {
      meta.setAttribute('content', isDark ? 'rgba(11, 17, 32, 0.5)' : 'rgba(248, 250, 252, 0.5)');
    }
  }, [isSidebarOpen, isInsightsOpen]);

  return (
    <div className="relative flex max-lg:min-h-[100dvh] lg:h-full w-full text-slate-900 dark:text-slate-100 max-lg:overflow-visible lg:overflow-hidden font-sans">
      {/* Background elements for depth */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-blue-500/10 rounded-full blur-[100px] pointer-events-none" />

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
      <div className="flex w-full max-lg:min-h-[100dvh] lg:h-full p-0 md:p-6 gap-0 md:gap-6 z-10 lg:pt-6">

        {/* Desktop Sidebar (Persistent) */}
        <Sidebar className="hidden lg:flex lg:w-[18%] lg:min-w-[240px]" />

        {/* Mobile Sidebar Drawer (Overlay) */}
        <AnimatePresence>
          {isSidebarOpen && (
            <>
              {/* Backdrop — 從安全區下方開始，不蓋靈動島 */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setSidebarOpen(false)}
                className="fixed left-0 right-0 bottom-0 bg-black/60 backdrop-blur-sm z-[90] lg:hidden"
                style={{ top: 'env(safe-area-inset-top)' }}
              />
              {/* Drawer — 背景從 inset-y-0 延伸，content 用 padding 避開安全區 */}
              <motion.aside
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="fixed inset-y-0 left-0 w-[280px] z-[100] lg:hidden shadow-2xl"
                style={{
                  backgroundColor: 'var(--bg-gradient-start)',
                  borderRight: '1px solid rgba(128,128,128,0.15)',
                  backdropFilter: 'blur(20px)',
                  WebkitBackdropFilter: 'blur(20px)'
                }}
              >
                <Sidebar
                  className="flex-1 !bg-transparent !border-0 pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)]"
                  onItemClick={() => setSidebarOpen(false)}
                />
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* Main Content Area */}
        <div className="flex-1 flex gap-0 md:gap-4 max-lg:min-h-[100dvh] lg:h-full min-w-0 relative">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
