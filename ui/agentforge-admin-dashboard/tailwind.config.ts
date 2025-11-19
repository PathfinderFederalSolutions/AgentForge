import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class", '[data-theme="night"]'],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        day: { bg: "#05080D", grid: "#0E1622", lines: "#0F2237", accent: "#00A39B", neon: "#CCFF00", text: "#D6E2F0" },
        night: { bg: "#000000", text: "#FF2B2B", dim: "#891616", grid: "#1a0000" }
      },
      boxShadow: {
        hud: "0 0 0 1px rgba(255,255,255,0.06), 0 8px 40px rgba(0,0,0,0.35)"
      },
      keyframes: {
        pulseSlow: { "0%,100%": { opacity: .45 }, "50%": { opacity: .9 } },
        scan: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(100%)" } }
      },
      animation: {
        pulseSlow: "pulseSlow 3s ease-in-out infinite",
        scan: "scan 4s linear infinite"
      }
    }
  },
  plugins: []
};
export default config;
