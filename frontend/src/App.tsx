import { useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { DashboardLayout } from './components/DashboardLayout';
import { HomePage } from './pages/HomePage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SettingsPage } from './pages/SettingsPage';
import { useSettingsStore, Theme } from './stores/settingsStore';

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
      } else {
        root.classList.add('light');
        root.classList.remove('dark');
      }
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

  // Global Scroll Reset to prevent programmatic layout shifts
  useEffect(() => {
    window.scrollTo(0, 0);
    const preventScroll = () => window.scrollTo(0, 0);
    window.addEventListener('scroll', preventScroll);
    return () => window.removeEventListener('scroll', preventScroll);
  }, []);

  return (
    <Routes>
      <Route path="/" element={<DashboardLayout />}>
        <Route index element={<HomePage />} />
        <Route path="documents" element={<DocumentsPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}

export default App;
