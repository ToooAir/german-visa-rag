import { Menu, X, LayoutPanelLeft, Sparkles } from 'lucide-react';
import { useTranslation } from '../translations';

interface MobileHeaderProps {
  onToggleSidebar: () => void;
  onToggleInsights: () => void;
  isSidebarOpen?: boolean;
  isInsightsActive?: boolean;
  showInsightsAction?: boolean;
}

export function MobileHeader({ onToggleSidebar, onToggleInsights, isSidebarOpen, isInsightsActive, showInsightsAction }: MobileHeaderProps) {
  const { t } = useTranslation();

  return (
    <header className="fixed top-0 left-0 right-0 h-[calc(4rem+env(safe-area-inset-top))] pt-[env(safe-area-inset-top)] bg-background/80 backdrop-blur-md border-b border-white/10 z-[80] px-4 flex items-center justify-between lg:hidden">
      <button
        onClick={onToggleSidebar}
        className="p-2 text-slate-400 hover:text-white transition-colors"
        aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center text-accent ring-1 ring-accent/30">
          <Sparkles size={18} />
        </div>
        <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-wide">
          {(t.personaName as string) || 'Visa Assistant'}
        </span>
      </div>

      {showInsightsAction ? (
        <button
          onClick={onToggleInsights}
          className={`p-2 transition-colors relative ${isInsightsActive ? 'text-accent' : 'text-slate-400 hover:text-white'}`}
          aria-label="Open insights"
        >
          <LayoutPanelLeft size={24} />
          {isInsightsActive && (
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_8px_rgba(var(--accent-rgb),0.5)]" />
          )}
        </button>
      ) : (
        <div className="w-10" /> // Placeholder for balance
      )}
    </header>
  );
}
