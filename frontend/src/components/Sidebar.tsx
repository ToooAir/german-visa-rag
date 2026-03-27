import { Home, FileText, Settings } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function Sidebar({ className = '', onItemClick }: { className?: string, onItemClick?: () => void }) {
  const { t } = useTranslation();

  return (
    <div className={`glass-panel flex flex-col h-full overflow-hidden ${className}`}>
      {/* Brand & Fixed Nav Section */}
      <div className="p-5 pb-2 shrink-0">
        <div className="flex items-center gap-3 px-2 mb-8 mt-2">
          <div className="p-1.5 bg-accent/15 rounded-xl ring-1 ring-accent/20">
            <img src="/logo.png" className="w-8 h-8 object-contain" alt="" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">VisaFlow DE</h1>
        </div>

        {/* Main Nav */}
        <nav className="flex flex-col gap-1.5 mb-5">
          <NavItem icon={<Home size={18} />} label={t.home} to="/" end onClick={onItemClick} />
          <NavItem icon={<FileText size={18} />} label={t.library} to="/documents" onClick={onItemClick} />
          <NavItem icon={<Settings size={18} />} label={t.settings} to="/settings" onClick={onItemClick} />
        </nav>
      </div>

      {/* Scrollable Sources Section */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-5 pt-0">
        <div className="mb-8">
          <h2 className="text-sm lg:text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider px-2 mb-3 leading-none">{t.recentSources}</h2>
          <div className="flex flex-col gap-2.5">
            {/* Pinned Official Sources */}
            {useChatStore.getState().pinnedSources.map((s, i) => (
              <SourceItem key={`pinned-${i}`} title={s.title} url={s.url} official />
            ))}

            {/* Separator if we have recent ones */}
            {useChatStore.getState().recentSources.length > 0 && (
              <div className="h-px bg-slate-200 dark:bg-slate-800 my-1 mx-2" />
            )}

            {/* Dynamic Recent Sources */}
            {useChatStore(state => state.recentSources).slice(0, 3).map((s, i) => (
              <SourceItem
                key={`recent-${i}`}
                title={s.title || (s.url.includes('bamf') ? 'BAMF' : s.url.split('/')[2])}
                url={s.url}
                favicon={s.authority === 'official' ? '🏛️' : '📄'}
                official={s.authority === 'official'}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Subcomponents
function NavItem({ icon, label, to, end = false, onClick }: { icon: React.ReactNode, label: string, to: string, end?: boolean, onClick?: () => void }) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onClick}
      className={({ isActive }) => `flex items-center gap-3.5 px-3 py-2.5 rounded-xl text-base lg:text-[15px] font-medium transition-colors ${
        isActive ? 'bg-accent/15 text-accent' : 'text-slate-600 dark:text-slate-300 hover:bg-slate-200/50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
      }`}
    >
      {icon}
      {label}
    </NavLink>
  );
}

function SourceItem({ title, url, favicon, official = false }: { title: string, url: string, favicon?: React.ReactNode, official?: boolean }) {
  const defaultFavicon = official ? '🏛️' : '📄';

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-slate-200/50 dark:hover:bg-slate-800/30 cursor-pointer transition-all hover:translate-x-1 group"
    >
      <div className="w-6 h-6 rounded bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-xs border border-slate-300 dark:border-slate-700 group-hover:border-accent/40 transition-colors">
        {favicon || defaultFavicon}
      </div>
      <span className="text-[15px] lg:text-sm text-slate-700 dark:text-slate-300 truncate flex-1 group-hover:text-accent transition-colors">{title}</span>
      {official && <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />}
    </a>
  );
}
