import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LOCALES, SOURCE_PAGES, README_PAGES, TRACKS, colabUrl, notebookPath, repositoryUrl, route } from './catalog.mjs';
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
  const metadata = {
    title: chinese ? '实用模型训练指南' : 'Practical model-training guides',
    description: chinese ? 'advai-notebooks 的中英文训练指南、配置与 Colab 入口。' : 'Training guides, configurations and Colab entry points for advai-notebooks.',
    template: 'splash',
    editUrl: editUrl('site/src/components/HomeContent.astro'),
    hero: {
      title: chinese ? '从 Notebook 开始，走完训练流程。' : 'From notebook to trained model.',
    },
  };
  // Starlight overrides render catalog data and original README examples on the homepage.
  return frontmatter(metadata);
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
  // Retire only our old generated copy; the README's original banner is untouched.
  await rm(resolve(SITE_ROOT, 'public/generated/readme-hero.png'), { force: true });
  await writeFile(manifestFile, JSON.stringify([...pages.keys()], null, 2) + '\n');
  console.log(`Generated ${pages.size} documentation pages from repository sources (no notebook execution).`);
  return pages;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await syncContent();
