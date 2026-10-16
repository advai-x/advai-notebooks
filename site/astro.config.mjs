import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { BASE_PATH, REPOSITORY, SITE_ORIGIN } from './scripts/catalog.mjs';

const group = (label, chinese, items) => ({
  label,
  translations: { 'zh-CN': chinese },
  items: items.map((slug) => ({ slug })),
});

export default defineConfig({
  site: SITE_ORIGIN,
  base: BASE_PATH,
  trailingSlash: 'always',
  output: 'static',
  integrations: [
    starlight({
      title: 'advai-notebooks',
      favicon: '/brand-mark.png',
      head: [{
        tag: 'link',
        attrs: { rel: 'icon', type: 'image/png', href: `${BASE_PATH}/brand-mark-dark.png`, media: '(prefers-color-scheme: dark)' },
      }],
      description: 'Practical, reproducible training guides for modern AI models.',
      defaultLocale: 'en',
      locales: {
        en: { label: 'English', lang: 'en' },
        'zh-cn': { label: '简体中文', lang: 'zh-CN' },
      },
      social: [{ icon: 'github', label: 'GitHub', href: REPOSITORY }],
      customCss: ['./src/styles/custom.css'],
      components: {
        Header: './src/components/Header.astro',
        SiteTitle: './src/components/BrandTitle.astro',
        Hero: './src/components/Hero.astro',
        MarkdownContent: './src/components/MarkdownContent.astro',
      },
      sidebar: [
        group('Start here', '从这里开始', ['start/quickstart', 'training', 'start/workflow']),
        group('Retrieval & decisions', '检索与决策', ['guides/embedding', 'guides/ranker', 'guides/decision']),
        group('Language models', '语言模型', ['guides/llm', 'guides/llm-moe', 'guides/rl']),
        group('Speech models', '语音模型', ['guides/vad', 'guides/stt', 'guides/tts']),
        group('Multimodal & generation', '多模态与生成', ['guides/diffusion', 'guides/vlm', 'guides/vla', 'guides/vm']),
        group('Data & publication', '数据与发布', ['reference/retrieval-data', 'reference/datasets', 'reference/publishing', 'reference/cli']),
        group('Development', '开发与贡献', ['reference/architecture', 'development/contributing', 'development/docs-site']),
      ],
    }),
  ],
});
