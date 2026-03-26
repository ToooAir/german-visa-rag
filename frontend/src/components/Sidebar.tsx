import { Home, FileText, Settings, Briefcase, FileCode2, BookOpen } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function Sidebar({ className = '' }: { className?: string }) {
  const { activeVisaCategory, setActiveVisaCategory } = useChatStore();
  const { t } = useTranslation();

  const handleCategoryClick = (category: string) => {
    setActiveVisaCategory(category);
  };

  return (
    <div className={`glass-panel flex flex-col p-4 h-full overflow-y-auto ${className}`}>
      <div className="flex items-center gap-3 px-2 mb-8">
        <div className="p-1.5 bg-accent/15 rounded-xl ring-1 ring-accent/20">
          <img src="/logo.png" className="w-8 h-8 object-contain" alt="" />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">VisaFlow DE</h1>
      </div>

      {/* Main Nav */}
      <nav className="flex flex-col gap-1 mb-8">
        <NavItem icon={<Home size={18} />} label={t.home} to="/" end />
        <NavItem icon={<FileText size={18} />} label={t.library} to="/documents" />
        <NavItem icon={<Settings size={18} />} label={t.settings} to="/settings" />
      </nav>

      {/* Recent Sources */}
      <div className="mb-8">
        <h2 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider px-2 mb-3">{t.recentSources}</h2>
        <div className="flex flex-col gap-2">
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

      {/* Visa Categories */}
      <div>
        <h2 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider px-2 mb-3">{t.visaCategories}</h2>
        <div className="flex flex-col gap-1">
          <CategoryItem 
            icon={<Briefcase size={16} />} 
            label={t.chancenkarte} 
            active={activeVisaCategory === 'chancenkarte'} 
            onClick={() => handleCategoryClick('chancenkarte')}
          />
          <CategoryItem 
            icon={<FileCode2 size={16} />} 
            label={t.skilledWorker} 
            active={activeVisaCategory === 'work_visa'} 
            onClick={() => handleCategoryClick('work_visa')}
          />
          <CategoryItem 
            icon={<FileText size={16} />} 
            label={t.blueCard} 
            active={activeVisaCategory === 'blue_card'} 
            onClick={() => handleCategoryClick('blue_card')}
          />
          <CategoryItem 
            icon={<BookOpen size={16} />} 
            label={t.studyVisa} 
            active={activeVisaCategory === 'student_visa'} 
            onClick={() => handleCategoryClick('student_visa')}
          />
        </div>
      </div>
    </div>
  );
}

// Subcomponents
function NavItem({ icon, label, to, end = false }: { icon: React.ReactNode, label: string, to: string, end?: boolean }) {
  return (
    <NavLink 
      to={to}
      end={end}
      className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
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
      <span className="text-sm text-slate-700 dark:text-slate-300 truncate flex-1 group-hover:text-accent transition-colors">{title}</span>
      {official && <div className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />}
    </a>
  );
}

function CategoryItem({ icon, label, active = false, onClick }: { icon: React.ReactNode, label: string, active?: boolean, onClick?: () => void }) {
  return (
    <button onClick={onClick} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
      active ? 'bg-slate-200/80 dark:bg-slate-800/80 text-accent border border-slate-300/50 dark:border-slate-700/50' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800/30'
    }`}>
      {icon}
      <span className="flex-1 text-left">{label}</span>
      {active && <div className="w-1.5 h-1.5 rounded-full bg-accent" />}
    </button>
  );
}
