/** @type {import('tailwindcss').Config} */
import tailwindcssSafeArea from 'tailwindcss-safe-area';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'var(--bg-gradient-start)',
        panel: 'var(--panel-bg)',
        panelBorder: 'var(--panel-border)',
        accent: '#06B6D4',
      }
    },
  },
  plugins: [
    tailwindcssSafeArea,
  ],
}
