import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(({command, mode}) => {
  // Base path for GitHub Pages deployment (https://likithaa2203.github.io/SentinelML/)
  // Supports VITE_BASE_PATH override, defaults to '/SentinelML/' for GitHub Pages builds,
  // and '/' for local dev / AI Studio preview.
  const isGitHubPages = process.env.GITHUB_PAGES === 'true' || process.env.VITE_DEPLOY_TARGET === 'gh-pages';
  const base = process.env.VITE_BASE_PATH || (isGitHubPages ? '/SentinelML/' : (process.env.APPLET_ID ? '/' : '/SentinelML/'));

  return {
    base,
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
