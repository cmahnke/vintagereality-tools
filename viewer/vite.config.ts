import { defineConfig } from 'vite';
import { resolve } from "path";
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { NodePackageImporter } from "sass";
import * as fs from 'fs';
import * as path from 'path';

const BASE_URL = process.env.BASE_URL || '';

function generateImagesJson() {
  return {
    name: 'generate-images-json',
    buildStart() {
      const imagesDir = path.resolve(__dirname, 'public/images');
      const outputPath = path.resolve(__dirname, 'public/images.json');
      
      if (fs.existsSync(imagesDir)) {
        const files = fs.readdirSync(imagesDir).filter(file => {
          return /\.(heic)$/i.test(file);
        });
        fs.writeFileSync(outputPath, JSON.stringify(files, null, 2));
        console.log(`Generated images.json with ${files.length} images.`);
      } else {
        console.warn('public/images directory not found, skipping images.json generation.');
      }
    }
  };
}

export default defineConfig({
  base: `${BASE_URL}/`,
  plugins: [
    generateImagesJson(),
    viteStaticCopy({
      targets: [
        /*
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
        */
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
      input: {
        main: resolve(__dirname, 'index.html'),
        images: resolve(__dirname, 'images.html')
      },
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