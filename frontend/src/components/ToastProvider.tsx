import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { useToastStore, ToastType } from '../stores/toastStore';

const icons: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle2 className="text-emerald-500" size={20} />,
  error: <AlertCircle className="text-red-500" size={20} />,
  warning: <AlertCircle className="text-amber-500" size={20} />,
  info: <Info className="text-blue-500" size={20} />
};

export function ToastProvider() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="fixed top-6 left-1/2 -translate-x-1/2 lg:left-auto lg:right-6 lg:translate-x-0 z-[100] flex flex-col gap-3 pointer-events-none w-[calc(100%-32px)] lg:w-auto max-w-[400px]">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            layout
            className="pointer-events-auto flex items-start gap-3 p-4 bg-white/95 dark:bg-slate-800/95 backdrop-blur-md border border-slate-200/50 dark:border-slate-700/50 shadow-2xl rounded-xl w-full"
          >
            <div className="shrink-0 mt-0.5">{icons[toast.type]}</div>
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">{toast.title}</h4>
              {toast.message && (
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                  {toast.message}
                </p>
              )}
            </div>
            <button
              onClick={() => removeToast(toast.id)}
              className="shrink-0 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors p-1"
            >
              <X size={14} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
