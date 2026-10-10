import { existsSync, statSync } from 'node:fs';
import { dirname, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkStringify from 'remark-stringify';
import { visit } from 'unist-util-visit';
import { REPOSITORY, SOURCE_PAGES, TRACKS, repositoryUrl, route } from './catalog.mjs';

export const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const SITE_ROOT = resolve(REPO_ROOT, 'site');
export const markdown = unified().use(remarkParse).use(remarkGfm).use(remarkStringify, {
  fences: true,
  bullet: '-',
  listItemIndent: 'one',
});

export function text(node) {
  return node.value ?? node.children?.map(text).join('') ?? '';
}

export function sourceMap() {
  const result = new Map();
  for (const page of SOURCE_PAGES) {
    for (const [locale, source] of Object.entries(page.sources)) {
      result.set(source, { slug: page.slug, locale });
    }
  }
  for (const track of TRACKS) {
    if (track.notebook) result.set(posix.join(posix.dirname(track.notebook), 'README.md'), { slug: track.guide, locale: 'en' });
  }
  result.set('README.md', { slug: '', locale: 'en' });
  result.set('README.zh-CN.md', { slug: '', locale: 'zh-cn' });
  result.set('notebooks/README.md', { slug: 'training', locale: 'en' });
  result.set('notebooks/README.zh-CN.md', { slug: 'training', locale: 'zh-cn' });
  result.set('notebooks/', { slug: 'training', locale: 'en' });
  return result;
}

export function resolveLink(url, source, locale, { root = REPO_ROOT, pages = sourceMap(), image = false } = {}) {
  if (!url || url.startsWith('#') || /^(?:[a-z][\w+.-]*:|\/\/)/i.test(url)) return url;
  const [, rawPath, suffix] = url.match(/^([^?#]*)(.*)$/);
  const target = posix.normalize(posix.join(rawPath.startsWith('/') ? '' : posix.dirname(source), decodeURIComponent(rawPath.replace(/^\/+/, ''))));
  if (target === '..' || target.startsWith('../') || target.startsWith('/')) {
    throw new Error(`Link escapes the repository: ${source} -> ${url}`);
  }
  const mapped = pages.get(target) ?? pages.get(`${target}/`);
  if (mapped && !image) {
    // Preserve explicit Chinese links; otherwise follow the reader's selected locale.
    return route(mapped.locale === 'zh-cn' ? 'zh-cn' : locale, mapped.slug) + suffix;
  }
  const absolute = resolve(root, target);
  if (!existsSync(absolute)) throw new Error(`Missing repository target: ${source} -> ${url}`);
  if (image) {
    return `https://raw.githubusercontent.com/advai-x/advai-notebooks/main/${target.split('/').map(encodeURIComponent).join('/')}${suffix}`;
  }
  const translated = target.replace(/(?<!\.zh-CN)\.ipynb$/, '.zh-CN.ipynb');
  const preferred = locale === 'zh-cn' && target.endsWith('.ipynb') && existsSync(resolve(root, translated)) ? translated : target;
  return repositoryUrl(preferred, statSync(absolute).isDirectory()) + suffix;
}

export function extractSections(input, titles) {
  const tree = markdown.parse(input);
  const selected = [];
  for (const title of titles) {
    const start = tree.children.findIndex((node) => node.type === 'heading' && node.depth === 2 && text(node) === title);
    if (start < 0) throw new Error(`README section not found: ${title}`);
    let end = start + 1;
    while (end < tree.children.length && !(tree.children[end].type === 'heading' && tree.children[end].depth <= 2)) end++;
    selected.push(...tree.children.slice(start, end));
  }
  if (titles.length === 1) {
    selected.shift();
    for (const node of selected) if (node.type === 'heading') node.depth--;
  }
  return markdown.stringify({ type: 'root', children: selected });
}

export function rewriteMarkdown(input, source, locale, options = {}) {
  const tree = markdown.parse(input);
  if (tree.children[0]?.type === 'heading' && tree.children[0].depth === 1) tree.children.shift();
  // Starlight's language selector replaces the source document's language-switch line.
  tree.children = tree.children.filter((node) => !(node.type === 'paragraph' && /^English\s*\|\s*简体中文$/.test(text(node))));
  visit(tree, (node) => {
    if (['link', 'image', 'definition'].includes(node.type)) {
      node.url = resolveLink(node.url, source, locale, { ...options, image: node.type === 'image' });
    }
    if (node.type === 'html') {
      node.value = node.value.replace(/((?:href|src)\s*=\s*)(["'])(.*?)\2/gi, (_, prefix, quote, url) =>
        prefix + quote + resolveLink(url, source, locale, { ...options, image: /^src/i.test(prefix) }) + quote);
    }
  });
  // Convert GitHub alerts to Starlight asides without touching fenced examples.
  visit(tree, 'blockquote', (node) => {
    const first = node.children[0]?.children?.[0];
    const marker = first?.type === 'text' && first.value.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\n|$)/);
    if (!marker) return;
    first.value = first.value.slice(marker[0].length);
    const type = { NOTE: 'note', TIP: 'tip', IMPORTANT: 'caution', WARNING: 'caution', CAUTION: 'danger' }[marker[1]];
    const body = markdown.stringify({ type: 'root', children: node.children }).trim();
    node.type = 'html';
    node.value = `:::${type}\n${body}\n:::`;
    delete node.children;
  });
  return markdown.stringify(tree);
}

export function frontmatter(data) {
  // JSON is a YAML subset; this safely quotes arbitrary bilingual titles and URLs.
  return `---\n${JSON.stringify(data, null, 2)}\n---\n\n`;
}

export function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
}

export function sourceCredit(source, locale) {
  return locale === 'en'
    ? `Source: [\`${source}\`](${repositoryUrl(source)}).`
    : `文档来源：[\`${source}\`](${repositoryUrl(source)})。`;
}

export const editUrl = (source) => `${REPOSITORY}/edit/main/${source}`;
