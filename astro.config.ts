import {defineConfig} from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://cbenge509.github.io',
  base: '/BERTVision/',
  output: 'static',
  vite: {
    plugins: [tailwindcss()],
  },
});
