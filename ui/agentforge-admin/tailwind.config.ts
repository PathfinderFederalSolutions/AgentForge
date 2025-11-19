import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class", '[data-theme="night"]'],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        day: { 
          bg: "#05080D", 
          grid: "#0E1622", 
          lines: "#0F2237", 
          accent: "#00A39B", 
          neon: "#CCFF00", 
          text: "#D6E2F0" 
        },
        night: { 
          bg: "#000000", 
          text: "#FF2B2B", 
          dim: "#891616", 
          grid: "#1a0000" 
        }
      },
      maxWidth: {
        '4xl': '56rem',
        '2xl': '42rem'
      },
      boxShadow: {
        hud: "0 0 0 1px rgba(255,255,255,0.06), 0 8px 40px rgba(0,0,0,0.35)",
        glow: "0 0 20px rgba(0, 163, 155, 0.3)",
        "glow-red": "0 0 20px rgba(255, 43, 43, 0.3)"
      },
      keyframes: {
        pulseSlow: { "0%,100%": { opacity: .45 }, "50%": { opacity: .9 } },
        scan: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(100%)" } },
        typing: { "0%": { opacity: "1" }, "50%": { opacity: "0" }, "100%": { opacity: "1" } },
        slideUp: { "0%": { transform: "translateY(20px)", opacity: "0" }, "100%": { transform: "translateY(0)", opacity: "1" } }
      },
      animation: {
        pulseSlow: "pulseSlow 3s ease-in-out infinite",
        scan: "scan 4s linear infinite",
        typing: "typing 1s ease-in-out infinite",
        slideUp: "slideUp 0.3s ease-out"
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', 'monospace']
      }
    }
  },
  plugins: []
};

export default config;
