import { Bell, Palette, Globe, Github, ExternalLink, Code2 } from 'lucide-react';
import { useSettingsStore, Theme, Language } from '../stores/settingsStore';
import { useTranslation } from '../translations';

export function SettingsPage() {
  const { theme, setTheme, language, setLanguage } = useSettingsStore();
  const { t } = useTranslation();

  return (
    <div className="flex-1 glass-panel px-7 pt-[calc(6rem+env(safe-area-inset-top))] pb-[calc(2rem+env(safe-area-inset-bottom))] sm:p-8 bg-slate-50/40 dark:bg-slate-900/40 lg:rounded-2xl rounded-none border-0 lg:border border-slate-200/50 dark:border-slate-800/50 lg:shadow-2xl max-lg:overflow-visible lg:overflow-y-auto">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100">{t.settings}</h2>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 text-accent">
          <Code2 size={12} />
          <span className="text-[10px] font-bold uppercase tracking-wider">v1.2.0 Open Source</span>
        </div>
      </div>

      <div className="max-w-2xl space-y-6">
        <section className="glass-panel p-6 bg-white/40 dark:bg-slate-800/40">
          <div className="flex items-center gap-3 mb-4 text-accent">
            <Palette size={20} />
            <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t.appearance}</h3>
          </div>
          <div className="flex items-center justify-between py-3 border-b border-slate-200/50 dark:border-slate-700/50">
            <span className="text-slate-700 dark:text-slate-300">{t.themePreference}</span>
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as Theme)}
              className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 text-sm rounded-lg px-3 py-1.5 outline-none cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              <option value="dark">{t.dark}</option>
              <option value="light">{t.light}</option>
              <option value="system">{t.system}</option>
            </select>
          </div>
        </section>

        <section className="glass-panel p-6 bg-white/40 dark:bg-slate-800/40">
          <div className="flex items-center gap-3 mb-4 text-accent">
            <Globe size={20} />
            <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t.langRegion}</h3>
          </div>
          <div className="flex items-center justify-between py-3 border-b border-slate-200/50 dark:border-slate-700/50">
            <span className="text-slate-700 dark:text-slate-300">{t.chatLang}</span>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 text-sm rounded-lg px-3 py-1.5 outline-none cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              <option value="en">English</option>
              <option value="de">Deutsch</option>
              <option value="zh-TW">繁體中文</option>
            </select>
          </div>
        </section>

        <section className="glass-panel p-6 bg-white/40 dark:bg-slate-800/40">
          <div className="flex flex-col gap-1 mb-4">
            <div className="flex items-center gap-3 text-accent">
              <Bell size={20} />
              <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t.smartAlerts || 'Smart Progress Alerts'}</h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 pl-8">{t.smartAlertsDesc || 'Receive dynamic alerts when AI detects milestone updates.'}</p>
          </div>
          <div className="flex items-center justify-between py-3">
            <span className="text-slate-700 dark:text-slate-300">{t.notifications || 'Enable Notifications'}</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useSettingsStore((state) => state.notificationsEnabled)}
                onChange={(e) => useSettingsStore.getState().setNotificationsEnabled(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-slate-200 dark:bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent border border-slate-300 dark:border-transparent"></div>
            </label>
          </div>
        </section>

        <section className="glass-panel p-6 bg-white/40 dark:bg-slate-800/40">
          <div className="flex items-center gap-3 mb-4 text-slate-900 dark:text-white">
            <Github size={20} />
            <h3 className="font-semibold">{t.sourceCode}</h3>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">{t.sourceCodeDesc}</p>
          <a
            href="https://github.com/toooair/german-visa-rag"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between p-4 rounded-xl bg-slate-100/50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 hover:border-accent group transition-all"
          >
            <span className="text-sm font-bold text-slate-700 dark:text-slate-200 group-hover:text-accent transition-colors">GitHub Repository</span>
            <ExternalLink size={16} className="text-slate-400 group-hover:text-accent transition-colors" />
          </a>
        </section>
      </div>
    </div>
  );
}
