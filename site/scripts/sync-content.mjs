import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { BASE_PATH, LOCALES, REPOSITORY, SOURCE_PAGES, README_PAGES, TRACKS, colabUrl, notebookPath, repositoryUrl, route } from './catalog.mjs';
import { REPO_ROOT, SITE_ROOT, editUrl, escapeHtml, extractSections, frontmatter, rewriteMarkdown, sourceCredit } from './content.mjs';

const outputDirectory = resolve(SITE_ROOT, 'src/content/docs');
const manifestFile = resolve(SITE_ROOT, '.content-manifest.json');

function actions(guide, locale) {
  const links = TRACKS.filter((track) => track.guide === guide).flatMap((track) => {
    const notebook = notebookPath(track, locale);
    return notebook
      ? [[locale === 'en' ? `${track.name} · Open in Colab` : `${track.name} · 在 Colab 中打开`, colabUrl(notebook)],
         [locale === 'en' ? `${track.name} · Notebook` : `${track.name} · Notebook 源文件`, repositoryUrl(notebook)]]
      : [[`${track.name} · ${locale === 'en' ? 'Configuration' : '配置'}`, repositoryUrl(track.config)]];
  });
  return links.length ? `<div class="guide-actions">${links.map(([label, href]) => `<a href="${escapeHtml(href)}">${escapeHtml(label)}</a>`).join('\n')}</div>\n\n` : '';
}

function fallbackNotice(source) {
  return `:::note[此页面暂未提供中文正文]\n下方保留英文原文。已有中文 Notebook 的训练路线，可从[训练目录](${route('zh-cn', 'training')})或本页入口打开。翻译就绪并加入文档目录后，构建会使用中文源文件。\n英文来源：[\`${source}\`](${repositoryUrl(source)})。\n:::\n\n`;
}

function overview(locale) {
  const chinese = locale === 'zh-cn';
  const rows = TRACKS.map((track) => {
    const notebook = notebookPath(track, locale);
    const start = notebook ? `[${chinese ? '在 Colab 中打开' : 'Open in Colab'}](${colabUrl(notebook)}) · [Notebook](${repositoryUrl(notebook)})` : (chinese ? 'CLI 指南；暂无 Notebook' : 'CLI guide; no notebook yet');
    return `| [${track.name}](${route(locale, track.guide)}) | ${track.description[locale]} | \`${track.extra}\` | ${start} |`;
  });
  const preface = chinese
    ? '先选择与你的训练目标最接近的路径。所有 smoke 配置使用随机 tiny 模型和合成数据，仅验证工程链路；真实模型仍需按数据和硬件验证。'
    : 'Choose the track closest to your training objective. Smoke configurations use tiny random models and synthetic data to validate the pipeline, not real model quality. Real checkpoints require validation on your data and hardware.';
  const workflowNotebook = `notebooks/workflows/01_retrieval_comparison${chinese ? '.zh-CN' : ''}.ipynb`;
  return `${preface}\n\n| ${chinese ? '路径 | 训练目标 | 安装 extra | 运行入口' : 'Track | Training objective | Install extra | Run'} |\n| --- | --- | --- | --- |\n${rows.join('\n')}\n\n## ${chinese ? 'Smoke 配置' : 'Smoke configurations'}\n\n${TRACKS.map((track) => `- ${track.name}: [\`${track.config}\`](${repositoryUrl(track.config)})`).join('\n')}\n\n## ${chinese ? '可选检索组合流程' : 'Optional retrieval workflow'}\n\n${chinese ? 'Embedding 和 Ranker 仍独立训练；此 Notebook 对比召回与重排结果。' : 'Embedding and ranker remain independently trainable. This notebook compares retrieval and reranking.'}\n\n[${chinese ? '在 Colab 中打开' : 'Open in Colab'}](${colabUrl(workflowNotebook)}) · [Notebook](${repositoryUrl(workflowNotebook)})\n`;
}

function homepage(locale) {
  const chinese = locale === 'zh-cn';
  const groups = chinese ? [
    ['检索与决策', 'Embedding、Ranker 与类型化 Decision 模型。', 'guides/embedding'],
    ['语言模型', '继续预训练、指令微调，以及独立的 PPO / GRPO。', 'guides/llm'],
    ['多模态与生成', 'Diffusion、图像问答、动作 chunk 与数值 World Model。', 'guides/vlm'],
    ['完整实验流程', '从 smoke 配置到自己的数据、评估、恢复与发布。', 'start/workflow'],
  ] : [
    ['Retrieval & decisions', 'Embedding encoders, rerankers and typed decision models.', 'guides/embedding'],
    ['Language models', 'Continued pre-training, instruction tuning and independent PPO / GRPO.', 'guides/llm'],
    ['Multimodal & generation', 'Diffusion, image QA, action chunks and numerical World Models.', 'guides/vlm'],
    ['The complete workflow', 'From a smoke configuration to your own data, evaluation, recovery and publication.', 'start/workflow'],
  ];
  const metadata = {
    title: chinese ? '实用模型训练指南' : 'Practical model-training guides',
    description: chinese ? 'advai-notebooks 的中英文训练指南、配置与 Colab 入口。' : 'Training guides, configurations and Colab entry points for advai-notebooks.',
    template: 'splash',
    editUrl: editUrl('site/scripts/sync-content.mjs'),
    hero: {
      title: chinese ? '从 Notebook 开始，走完训练流程。' : 'From notebook to trained model.',
      tagline: chinese ? '实用、可复现的训练指南：先完成 tiny smoke 实验，再使用自己的数据。' : 'Practical, reproducible training guides. Start with a tiny smoke run, then move to your own data.',
      image: { html: `<img src="${BASE_PATH}/generated/readme-hero.png" alt="advai-notebooks" width="2172" height="724" loading="eager" decoding="async">` },
      actions: [
        { text: chinese ? '快速开始' : 'Quick start', link: route(locale, 'start/quickstart'), icon: 'right-arrow' },
        { text: chinese ? '选择训练指南' : 'Choose a guide', link: route(locale, 'training'), variant: 'secondary' },
      ],
    },
  };
  const cards = `<div class="track-grid">\n${groups.map(([title, description, slug]) => `<a class="track-card" href="${route(locale, slug)}"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(description)}</span></a>`).join('\n')}\n</div>`;
  const readme = chinese ? 'README.zh-CN.md' : 'README.md';
  const caveat = chinese
    ? ':::caution[先验证链路，再判断模型质量]\nTiny 模型和合成数据只验证工程流程。VLA / VM 是训练基线，不代表已验证的机器人控制或规划能力。大型真实基座不保证适用于免费 Colab。\n:::'
    : ':::caution[Validate the pipeline before judging quality]\nTiny models and synthetic data validate the engineering workflow only. VLA / VM are trainable baselines, not demonstrated robotics or planning capabilities. Large real checkpoints are not guaranteed to fit free Colab runtimes.\n:::';
  const body = `## ${chinese ? '选择你的起点' : 'Choose your starting point'}\n\n${cards}\n\n${caveat}\n\n## ${chinese ? '同一份实现，多种运行方式' : 'One implementation, multiple ways to learn'}\n\n${chinese ? '文档用于阅读，Notebook 用于交互学习，CLI 用于可配置实验。三者调用同一套训练实现。' : 'Read the guides, learn interactively in notebooks, or run configurable experiments through the CLI. The notebooks and CLI share the same training implementation.'}\n\n[${chinese ? '查看全部 11 条训练路线' : 'Browse all 11 training tracks'}](${route(locale, 'training')}) · [GitHub](${REPOSITORY}) · [${chinese ? '项目介绍' : 'Project overview'}](${repositoryUrl(readme)})\n`;
  return frontmatter(metadata) + body;
}

export async function syncContent({ write = true } = {}) {
  const pages = new Map();
  const readmes = Object.fromEntries(await Promise.all(LOCALES.map(async (locale) => [locale, await readFile(resolve(REPO_ROOT, locale === 'en' ? 'README.md' : 'README.zh-CN.md'), 'utf8')])));
  for (const locale of LOCALES) {
    pages.set(`${locale}/index.md`, homepage(locale));
    pages.set(`${locale}/training.md`, frontmatter({ title: locale === 'en' ? 'Choose a training guide' : '选择训练指南', editUrl: editUrl('site/scripts/catalog.mjs') }) + overview(locale));
    pages.set(`${locale}/404.md`, frontmatter({ title: locale === 'en' ? 'Page not found' : '找不到页面', editUrl: false, pagefind: false })
      + (locale === 'en' ? `This page could not be found. [Go to the documentation home](${route(locale)}) or [choose a training guide](${route(locale, 'training')}).\n` : `此页面不存在。请返回[文档首页](${route(locale)})，或打开[训练目录](${route(locale, 'training')})。\n`));
    for (const page of SOURCE_PAGES) {
      const source = page.sources[locale] ?? page.sources.en;
      const input = await readFile(resolve(REPO_ROOT, source), 'utf8');
      const body = rewriteMarkdown(input, source, locale);
      const fallback = !page.sources[locale];
      pages.set(`${locale}/${page.slug}.md`, frontmatter({ title: page.title[locale], editUrl: editUrl(source) })
        + actions(page.slug, locale) + (fallback ? fallbackNotice(source) + '<section lang="en">\n\n' : '')
        + body + (fallback ? '\n</section>\n' : '') + '\n' + sourceCredit(source, locale) + '\n');
    }
    for (const page of README_PAGES) {
      const source = locale === 'en' ? 'README.md' : 'README.zh-CN.md';
      const excerpt = extractSections(readmes[locale], page.sections[locale]);
      const body = rewriteMarkdown((page.intro?.[locale] ? page.intro[locale] + '\n\n' : '') + excerpt, source, locale);
      pages.set(`${locale}/${page.slug}.md`, frontmatter({ title: page.title[locale], editUrl: editUrl(source) }) + body + '\n' + sourceCredit(source, locale) + '\n');
    }
  }

  // Starlight uses this content for GitHub Pages' special, non-localized 404.html.
  pages.set('404.md', frontmatter({ title: 'Page not found / 找不到页面', editUrl: false, pagefind: false })
    + `This page could not be found. / 此页面不存在。\n\n[English documentation](${route('en')}) · [中文文档](${route('zh-cn')})\n`);
  if (!write) return pages;

  // Delete only stale files named in our own previous generation manifest, never a source tree.
  let previous = [];
  try { previous = JSON.parse(await readFile(manifestFile, 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  for (const file of previous) {
    const absolute = resolve(outputDirectory, file);
    const relativePath = relative(outputDirectory, absolute);
    if (relativePath.startsWith(`..${sep}`) || relativePath === '..' || !file.endsWith('.md')) throw new Error(`Unsafe generated path: ${file}`);
    if (!pages.has(file)) await rm(absolute, { force: true });
  }
  for (const [file, contents] of pages) {
    const destination = resolve(outputDirectory, file);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, contents);
  }
  const assets = resolve(SITE_ROOT, 'public/generated');
  await mkdir(assets, { recursive: true });
  await copyFile(resolve(REPO_ROOT, 'docs/assets/readme-hero.png'), resolve(assets, 'readme-hero.png'));
  await writeFile(manifestFile, JSON.stringify([...pages.keys()], null, 2) + '\n');
  console.log(`Generated ${pages.size} documentation pages from repository sources (no notebook execution).`);
  return pages;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await syncContent();
