import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// Separate build for the embed loader: a tiny dependency-free IIFE that site owners
// reference via <script src>. It carries NO COEP and must NOT pull in the app — it
// only imports the WidgetConfig *type* (erased at build). `emptyOutDir: false` so it
// appends `loader.js` to the main `dist/` instead of wiping it.
export default defineConfig({
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    emptyOutDir: false,
    target: 'es2019',
    lib: {
      entry: fileURLToPath(new URL('./src/embed/loader.ts', import.meta.url)),
      name: 'AidekinLoader',
      formats: ['iife'],
      fileName: () => 'loader.js',
    },
    rollupOptions: {
      output: { entryFileNames: 'loader.js', extend: true },
    },
  },
})
