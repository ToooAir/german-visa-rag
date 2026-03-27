import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './components/DashboardLayout';
import { HomePage } from './pages/HomePage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SettingsPage } from './pages/SettingsPage';
import { useSettingsStore, Theme } from './stores/settingsStore';
import { useChatStore } from './stores/chatStore';
import { translations } from './translations';
import { ToastProvider } from './components/ToastProvider';

function App() {
  const { theme } = useSettingsStore();

  useEffect(() => {
    const root = window.document.documentElement;

    const applyTheme = (currentTheme: Theme) => {
      const isDark = currentTheme === 'dark' ||
        (currentTheme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

      console.log(`[Theme] Applying theme: ${currentTheme}, IsDark: ${isDark}`);

      if (isDark) {
        root.classList.add('dark');
        root.classList.remove('light');
        updateThemeColor('#0B1120');
      } else {
        root.classList.add('light');
        root.classList.remove('dark');
        updateThemeColor('#F8FAFC');
      }
    };

    const updateThemeColor = (color: string) => {
      let meta = document.querySelector('meta[name="theme-color"]');
      if (!meta) {
        meta = document.createElement('meta');
        meta.setAttribute('name', 'theme-color');
        document.head.appendChild(meta);
      }
      meta.setAttribute('content', color);
    };

    applyTheme(theme);

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      if (theme === 'system') {
        console.log(`[Theme] System theme changed. IsDark: ${e.matches}`);
        applyTheme('system');
      }
    };

    // Modern browsers
    mediaQuery.addEventListener('change', handleChange as EventListener);

    // Fallback for older browsers
    if (mediaQuery.addListener) {
      mediaQuery.addListener(handleChange);
    }

    return () => {
      mediaQuery.removeEventListener('change', handleChange as EventListener);
      if (mediaQuery.removeListener) {
        mediaQuery.removeListener(handleChange);
      }
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
