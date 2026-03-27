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
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setInsightsOpen(false)}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] lg:hidden"
            />
            {/* Drawer */}
            <motion.aside
              initial={{ x: 320 }}
              animate={{ x: 0 }}
              exit={{ x: 320 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 right-0 w-[320px] bg-background z-[70] lg:hidden shadow-2xl border-l border-white/10 pt-16"
            >
              <div className="flex flex-col h-full ring-1 ring-white/5">
                <div className="p-4 border-b border-white/5 flex items-center justify-center">
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
