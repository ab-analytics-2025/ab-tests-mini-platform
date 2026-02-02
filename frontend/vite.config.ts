import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// If you want to avoid CORS entirely, you can proxy the backend via Vite.
// - Set VITE_API_BASE="/api" in .env.local
// - Uncomment proxy below

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // proxy: {
    //   "/api": {
    //     target: "http://localhost:8080",
    //     changeOrigin: true,
    //     rewrite: (p) => p.replace(/^\/api/, ""),
    //   },
    // },
  },
});
