import { motion, AnimatePresence } from 'framer-motion';
import { ChatArea } from '../components/ChatArea';
import { InsightsPanel } from '../components/InsightsPanel';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function HomePage() {
  const { t } = useTranslation();
  const { isInsightsOpen, setInsightsOpen } = useChatStore();

  return (
    <>
      <ChatArea className="flex-1 min-w-0" />

      {/* Desktop InsightsPanel (Persistent) */}
      <InsightsPanel className="hidden lg:flex w-[25%] min-w-[320px]" />

      {/* Mobile Insights Drawer (Overlay) */}
      <AnimatePresence>
        {isInsightsOpen && (
          <>
            {/* Backdrop — 同樣從安全區下方開始 */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setInsightsOpen(false)}
              className="fixed left-0 right-0 bottom-0 bg-black/60 backdrop-blur-sm z-[90] lg:hidden"
              style={{ top: 'env(safe-area-inset-top)' }}
            />
            {/* Drawer — 背景填到頂 */}
            <motion.aside
              initial={{ x: 320 }}
              animate={{ x: 0 }}
              exit={{ x: 320 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 right-0 w-[320px] z-[100] lg:hidden shadow-2xl"
              style={{
                backgroundColor: 'var(--bg-gradient-start)',
                borderLeft: '1px solid rgba(128,128,128,0.15)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)'
              }}
            >
              <div className="flex flex-col h-full pt-[calc(env(safe-area-inset-top)+64px)] pb-[env(safe-area-inset-bottom)]">
                <div className="px-4 pt-4 pb-3 border-b border-white/5 flex items-center justify-center">
                  <h3 className="font-semibold text-slate-100 uppercase tracking-wider text-xs opacity-70">
                    {t.progressSummary}
                  </h3>
                </div>
                <InsightsPanel className="flex-1 !bg-transparent !border-0" />
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
