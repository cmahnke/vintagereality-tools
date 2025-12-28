import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { NodePackageImporter } from "sass";

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: '.'
        },
        {
          src: 'node_modules/@6over3/zeroperl-ts/dist/esm/*.wasm',
          dest: '.'
        },
        {
          src: 'node_modules/libheif-js/libheif-wasm/*.wasm',
          dest: '.'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/*.mjs',
          dest: '.'
        },
        {
          src: '../weights/vintagereality-*.onnx',
          dest: 'model',
          rename: 'vintagereality.onnx'
        }
      ]
    })
  ],
  build: {
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`
      }
    }
  },
  css: {
    preprocessorOptions: {
      scss: {
        api: "modern-compiler",
        importers: [new NodePackageImporter()]
      }
    }
  }
})