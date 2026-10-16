import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import { visit } from 'unist-util-visit';
import { LOCALES, README_PAGES, SOURCE_PAGES, TRACKS, colabUrl, notebookPath, route } from '../scripts/catalog.mjs';
import { REPO_ROOT, extractSections, frontmatter, markdown, resolveLink, rewriteMarkdown } from '../scripts/content.mjs';
import { syncContent } from '../scripts/sync-content.mjs';

function examples(input) {
  const result = [];
  visit(markdown.parse(input), (node) => {
    // CommonMark renders line endings inside code spans as spaces, unlike fenced code.
    if (node.type === 'code' || node.type === 'inlineCode') result.push([node.type, node.lang ?? '', node.type === 'inlineCode' ? node.value.replace(/\r?\n/g, ' ') : node.value]);
  });
  return result;
}

test('all catalog entries and localized notebook/configuration paths exist', () => {
  for (const page of SOURCE_PAGES) {
    for (const source of Object.values(page.sources)) assert.ok(existsSync(resolve(REPO_ROOT, source)), source);
  }
  for (const track of TRACKS) {
    assert.ok(existsSync(resolve(REPO_ROOT, track.config)), track.config);
    for (const locale of LOCALES) {
      const notebook = notebookPath(track, locale);
      if (notebook) assert.ok(existsSync(resolve(REPO_ROOT, notebook)), notebook);
    }
  }
  assert.equal(TRACKS.filter((track) => !track.notebook).length, 2);
});

test('published links retain base path, locale, query and fragment', () => {
  assert.equal(resolveLink('publishing.md#upload-explicitly', 'docs/llm.md', 'en'), route('en', 'reference/publishing') + '#upload-explicitly');
  assert.equal(resolveLink('llm.md?example=1#data-contracts', 'docs/rl.zh-CN.md', 'zh-cn'), route('zh-cn', 'guides/llm') + '?example=1#data-contracts');
  assert.equal(resolveLink('llm.zh-CN.md', 'docs/rl.md', 'en'), route('zh-cn', 'guides/llm'));
  assert.equal(resolveLink('/docs/llm.md', 'docs/rl.md', 'en'), route('en', 'guides/llm'));
  assert.equal(resolveLink('../README.md', 'docs/llm.md', 'en'), route('en'));
  assert.equal(resolveLink('#data-contracts', 'docs/llm.md', 'en'), '#data-contracts');
});

test('unpublished targets use GitHub and preserve translated notebooks', () => {
  assert.match(resolveLink('../configs/vlm/hf.yaml', 'docs/vlm.md', 'en'), /github\.com\/advai-x\/advai-notebooks\/blob\/main\/configs\/vlm\/hf\.yaml$/);
  assert.match(resolveLink('dataset-cards/', 'docs/datasets.md', 'en'), /\/tree\/main\/docs\/dataset-cards\/?$/);
  assert.match(resolveLink('../notebooks/vlm/01_finetune.ipynb', 'docs/vlm.md', 'zh-cn'), /01_finetune\.zh-CN\.ipynb$/);
  assert.equal(resolveLink('https://example.com/model?a=1', 'docs/llm.md', 'en'), 'https://example.com/model?a=1');
});

test('MoE is a bilingual architecture guide with companion notebooks, not another task', async () => {
  const guide = SOURCE_PAGES.find((page) => page.slug === 'guides/llm-moe');
  assert.deepEqual(Object.keys(guide.sources).sort(), ['en', 'zh-cn']);
  assert.ok(!TRACKS.some((track) => track.guide === 'guides/llm-moe'));
  for (const locale of LOCALES) {
    const source = guide.sources[locale];
    const text = await readFile(resolve(REPO_ROOT, source), 'utf8');
    const output = rewriteMarkdown(text, source, locale);
    assert.match(output, /02_moe_finetune/);
    assert.equal(resolveLink('llm-moe.md', 'docs/llm.md', locale), route(locale, 'guides/llm-moe'));
  }
});

test('missing targets and paths escaping the repository fail loudly', () => {
  assert.throws(() => resolveLink('missing.md', 'docs/llm.md', 'en'), /Missing repository target/);
  assert.throws(() => resolveLink('../../../../.env', 'docs/llm.md', 'en'), /escapes the repository/);
  assert.throws(() => resolveLink('%2e%2e/%2e%2e/.env', 'docs/llm.md', 'en'), /escapes the repository/);
  assert.throws(() => route('unknown'), /Unsupported locale/);
});

test('Markdown links, reference definitions and raw HTML attributes are transformed', () => {
  const input = '# Guide\n\nEnglish | [简体中文](llm.zh-CN.md)\n\n[Data][contract]\n\n[contract]: datasets.md\n\n<a href="publishing.md">Publish</a>\n';
  const output = rewriteMarkdown(input, 'docs/llm.md', 'en');
  assert.ok(!output.includes('# Guide'));
  assert.ok(!output.includes('English |'));
  assert.ok(output.includes(route('en', 'reference/datasets')));
  assert.ok(output.includes(route('en', 'reference/publishing')));
});

test('code examples containing fake Markdown links are untouched', () => {
  const input = '# Guide\n\n`[not a link](missing.md)`\n\n```python\nurl = "[example](missing.md)"\nprint("English | 简体中文")\n```\n';
  assert.deepEqual(examples(rewriteMarkdown(input, 'docs/llm.md', 'en')), examples(input));
});

test('GitHub alerts become reader-facing asides', () => {
  const output = rewriteMarkdown('> [!IMPORTANT]\n> Synthetic metrics are not real model quality.\n', 'README.md', 'en');
  assert.match(output, /:::caution/);
  assert.ok(!output.includes('[!IMPORTANT]'));
});

test('README section extraction ignores headings inside fenced code', () => {
  const input = '## First\n\n### Actual subsection\n\n```text\n## Second\n```\n\n## Second\n\nReal body.\n';
  const extracted = extractSections(input, ['First']);
  assert.ok(extracted.includes('## Actual subsection'));
  assert.ok(extracted.includes('## Second\n```'));
  assert.ok(!extracted.includes('Real body.'));
  assert.throws(() => extractSections(input, ['Missing']), /section not found/);
});

test('all real imported guides keep their code examples', async () => {
  for (const page of SOURCE_PAGES) {
    for (const [locale, source] of Object.entries(page.sources)) {
      const input = await readFile(resolve(REPO_ROOT, source), 'utf8');
      assert.deepEqual(examples(rewriteMarkdown(input, source, locale)), examples(input), source);
    }
  }
});

test('generation produces paired routes, explicit fallbacks and Colab entry points', async () => {
  const pages = await syncContent({ write: false });
  assert.equal(pages.size, 2 * (3 + SOURCE_PAGES.length + README_PAGES.length) + 1);
  for (const locale of LOCALES) {
    for (const page of [...SOURCE_PAGES, ...README_PAGES]) assert.ok(pages.has(`${locale}/${page.slug}.md`));
    for (const track of TRACKS) {
      const notebook = notebookPath(track, locale);
      if (notebook) assert.ok(pages.get(`${locale}/training.md`).includes(colabUrl(notebook)));
    }
  }
  const fallback = pages.get('zh-cn/guides/embedding.md');
  assert.ok(fallback.includes('此页面暂未提供中文正文'));
  assert.ok(fallback.includes('<section lang="en">'));
  assert.ok(fallback.includes('01_finetune.zh-CN.ipynb'));
  assert.ok(!pages.get('zh-cn/guides/vlm.md').includes('此页面暂未提供中文正文'));
  assert.ok(pages.get('en/guides/vlm.md').includes('edit/main/docs/vlm.md'));
  assert.ok(!pages.get('en/start/quickstart.md').includes('table above'));
  assert.ok(!pages.get('zh-cn/start/quickstart.md').includes('从上表'));
});

test('homepages omit edit links while reading pages retain their source edit links', async () => {
  const pages = await syncContent({ write: false });
  const metadata = (path) => JSON.parse(pages.get(path).split('---\n')[1]);
  for (const locale of LOCALES) {
    assert.equal(metadata(`${locale}/index.md`).editUrl, false);
    for (const slug of ['training', ...SOURCE_PAGES.map((page) => page.slug), ...README_PAGES.map((page) => page.slug)]) {
      assert.match(metadata(`${locale}/${slug}.md`).editUrl, /^https:\/\/github\.com\/advai-x\/advai-notebooks\/edit\/main\//);
    }
  }
});

test('localized contribution pages publish MIT licensing with the repository license link', async () => {
  const pages = await syncContent({ write: false });
  for (const locale of LOCALES) {
    const page = pages.get(`${locale}/development/contributing.md`);
    assert.match(page, /\[MIT (?:License|许可证)\]\(https:\/\/github\.com\/advai-x\/advai-notebooks\/blob\/main\/LICENSE\)/);
    assert.doesNotMatch(page, /A project license has not been selected yet|项目尚未选择许可证/);
  }
});

test('frontmatter safely quotes multilingual metadata', () => {
  const title = 'A "quoted" title: 中文';
  const serialized = frontmatter({ title });
  assert.equal(JSON.parse(serialized.split('---\n')[1]).title, title);
});
