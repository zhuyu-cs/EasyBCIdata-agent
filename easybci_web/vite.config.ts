import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

const GATEWAY_URL = process.env.EASYBCI_GATEWAY_URL ?? "http://127.0.0.1:8642";
const DASHBOARD_URL = process.env.EASYBCI_DASHBOARD_URL ?? "http://127.0.0.1:9119";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "../easybci_cli/web_dist",
    emptyOutDir: true,
    // The only remaining >500KB chunks are shiki's on-demand language grammars
    // (loaded lazily when a code block of that language first renders), not the
    // first-paint bundle. Raise the warning ceiling so the build log isn't noisy
    // about chunks that are intentionally lazy.
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        // Split the stable, always-loaded vendor libs into their own chunks so
        // they cache independently of app code across releases and download in
        // parallel with the app bundle. We deliberately do NOT touch shiki:
        // it's dynamically imported (`import("shiki")`) and already splits its
        // language grammars into on-demand chunks — bundling it here would pull
        // several MB of grammars into the first paint. Markdown libs are grouped
        // since they always load with the conversation view.
        // (B12 — conservative split only; the shiki-in-Worker move stays
        // deferred per plan: localhost single-user, no measured slowness.)
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("shiki") || id.includes("@shikijs")) return undefined;
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) {
            return "react-vendor";
          }
          if (
            id.includes("react-markdown") || id.includes("remark") || id.includes("micromark") ||
            id.includes("mdast") || id.includes("unist") || id.includes("hast") ||
            id.includes("vfile") || id.includes("unified") || id.includes("decode-named-character-reference") ||
            id.includes("property-information") || id.includes("space-separated-tokens") ||
            id.includes("comma-separated-tokens")
          ) {
            return "markdown-vendor";
          }
          return undefined;
        },
      },
    },
  },
  server: {
    proxy: {
      "/v1": {
        target: GATEWAY_URL,
      },
      "/api": {
        target: DASHBOARD_URL,
        ws: true,
      },
    },
  },
});
