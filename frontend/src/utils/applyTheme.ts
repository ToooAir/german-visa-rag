import type { Theme } from '../stores/settingsStore';

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  const isDark =
    theme === 'dark' ||
    (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  const bgColor = isDark ? 'rgba(11, 17, 32, 0.5)' : 'rgba(248, 250, 252, 0.5)';

  // 1. Toggle Tailwind class
  root.classList.toggle('dark', isDark);
  root.classList.toggle('light', !isDark);

  // 同步更新 html.style.backgroundColor 防止 theme 切換後的 overscroll 顏色過時
  root.style.backgroundColor = isDark ? '#0B1120' : '#f8fafc';

  // 2. 移除 + 重新插入 theme-color meta，協助一般瀏覽器判斷
  const existing = document.querySelector('meta[name="theme-color"]');
  if (existing) existing.remove();

  const meta = document.createElement('meta');
  meta.setAttribute('name', 'theme-color');
  meta.setAttribute('content', bgColor);
  document.head.appendChild(meta);

  // 3. 核心修正：觸發 iOS GPU 合成層重新掃描
  //    使用 sidebar 相同的原理：backdrop-filter 會強制建立新的 GPU Compositing Layer，
  //    此時 iOS Safari 會重新掃描 viewport 邊緣顏色並同步靈動島區域。
  const trigger = document.createElement('div');
  trigger.style.cssText = [
    'position:fixed',
    'top:0',
    'left:0',
    'right:0',
    'bottom:0',
    'z-index:2147483647',
    'pointer-events:none',
    'background-color:transparent',
    '-webkit-backdrop-filter:blur(0.01px)',
    'backdrop-filter:blur(0.01px)',
    'transition:none',
  ].join(';');
  document.body.appendChild(trigger);

  // 透過雙 rAF 確保渲染管線執行過該 GPU 操作後再行移除，視覺完全無感
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (trigger.parentNode) {
        trigger.parentNode.removeChild(trigger);
      }
    });
  });
}
