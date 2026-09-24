import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Proxies /api to the FastAPI backend in dev, so the frontend can call a
// same-origin /api path and never has to think about CORS locally.
//
// The target is overridable via VITE_PROXY_TARGET because "localhost" means
// something different depending on how this is run: plain `npm run dev` on the
// host reaches the backend at localhost:8000, but inside the frontend's Docker
// container (docker-compose.yml), "localhost" refers to that container itself
// — the backend is a separate container, reachable by its service name
// (`http://backend:8000`) instead. docker-compose.yml sets this env var; plain
// `npm run dev` falls back to the localhost default.
const proxyTarget = process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [vue()],
  // maplibre-gl bundles its own web worker (maplibre-gl-worker.mjs). Vite's
  // esbuild-based dependency pre-bundling (optimizeDeps) doesn't handle that
  // correctly — the worker file goes missing from the optimized deps cache,
  // which silently breaks vector-tile parsing (the map still renders its
  // style's flat background color, but no actual tile data — this is exactly
  // the "map is blue and empty" symptom). Excluding it from optimizeDeps makes
  // Vite serve it unbundled instead, which works correctly with its worker.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
  server: {
    host: true, // listen on 0.0.0.0 too — needed so the container's port mapping can reach it
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
