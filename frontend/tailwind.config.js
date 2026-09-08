/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#2e1065',
        },
        surface: {
          base: '#ffffff',
          light: '#f8fafc',
          muted: '#f1f5f9',
          border: '#e2e8f0',
          hover: '#f8fafc',
        },
        clinical: {
          purple: '#6d28d9',
          violet: '#7c3aed',
          teal: '#0d9488',
          cyan: '#0891b2',
          pink: '#db2777',
          rose: '#e11d48',
          emerald: '#059669',
          amber: '#d97706',
        }
      },
      borderRadius: {
        'card': '18px',
        'btn': '11px',
        'upload': '16px',
      },
      boxShadow: {
        'soft': '0 2px 10px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0, 0, 0, 0.02)',
        'card': '0 4px 20px -2px rgba(99, 102, 241, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03)',
        'card-hover': '0 12px 28px -4px rgba(109, 40, 217, 0.12), 0 4px 10px -2px rgba(0, 0, 0, 0.04)',
        'card-premium': '0 10px 30px -5px rgba(15, 23, 42, 0.06), 0 4px 12px -2px rgba(15, 23, 42, 0.03)',
        'glow-purple': '0 0 24px rgba(124, 58, 237, 0.28)',
        'glow-teal': '0 0 24px rgba(13, 148, 136, 0.28)',
        'glow-rose': '0 0 24px rgba(225, 29, 72, 0.28)',
        'glow-indigo': '0 0 24px rgba(99, 102, 241, 0.28)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.07)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
