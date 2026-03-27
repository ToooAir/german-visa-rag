import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './components/DashboardLayout';
import { HomePage } from './pages/HomePage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SettingsPage } from './pages/SettingsPage';
import { useSettingsStore } from './stores/settingsStore';
import { useChatStore } from './stores/chatStore';
import { translations } from './translations';
import { ToastProvider } from './components/ToastProvider';

import { applyTheme } from './utils/applyTheme';

function App() {
  const { theme } = useSettingsStore();

  useEffect(() => {
    // 1. 初始化（頁面載入時同步一次，確保內容與 localStorage 一致）
    applyTheme(theme);

    // 2. 只在 system 模式時監聽系統主題變化
    if (theme !== 'system') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => applyTheme('system');

    // Modern browsers
    mediaQuery.addEventListener('change', handleChange);

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, [theme]);

  // Initialize Chat State (Symmetry & Selection Fix)
  useEffect(() => {
    const { messages, activeVisaCategory, resetProgress, checklist } = useChatStore.getState();
    const { language } = useSettingsStore.getState();
    const t = translations[language as keyof typeof translations] || translations.en;

    // 1. Localize welcome message if it's the English default and we are in another language
    if (
      messages.length === 1 &&
      messages[0].id === 'welcome' &&
      messages[0].content === translations.en.welcome &&
      language !== 'en'
    ) {
      useChatStore.setState((state) => ({
        messages: [{ ...state.messages[0], content: t.welcome }]
      }));
    }

    // 2. Active selection on fresh start
    // If we have no checklist items, it means resetProgress hasn't been called yet
    if (checklist.length === 0 && activeVisaCategory) {
      resetProgress();
    }
  }, []);

  return (
    <>
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<HomePage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
      <ToastProvider />
    </>
  );
}

export default App;
