import { Bell, Palette, Globe } from 'lucide-react';
import { useSettingsStore, Theme, Language } from '../stores/settingsStore';
import { useTranslation } from '../translations';

export function SettingsPage() {
  const { theme, setTheme, language, setLanguage } = useSettingsStore();
  const { t } = useTranslation();

  return (
    <div className="flex-1 glass-panel p-8 bg-slate-50/40 dark:bg-slate-900/40 rounded-2xl border border-slate-200/50 dark:border-slate-800/50 shadow-2xl overflow-y-auto">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-8">{t.settings}</h2>
      
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
          <div className="flex items-center gap-3 mb-4 text-accent">
            <Bell size={20} />
            <h3 className="font-semibold text-slate-800 dark:text-slate-200">{t.notifications}</h3>
          </div>
          <div className="flex items-center justify-between py-3">
            <span className="text-slate-700 dark:text-slate-300">{t.emailAlerts}</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" defaultChecked className="sr-only peer" />
              <div className="w-11 h-6 bg-slate-200 dark:bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent border border-slate-300 dark:border-transparent"></div>
            </label>
          </div>
        </section>
      </div>
    </div>
  );
}
