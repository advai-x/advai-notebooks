import { access, readdir, readFile } from 'node:fs/promises';
import { relative, resolve, sep } from 'node:path';
import { load } from 'cheerio';
import { BASE_PATH, LOCALES, README_PAGES, REPOSITORY, SITE_ORIGIN, SOURCE_PAGES, TRACKS, colabUrl, notebookPath, route } from './catalog.mjs';
import { SITE_ROOT } from './content.mjs';
import { notebookCount } from './landing.mjs';

const dist = resolve(SITE_ROOT, 'dist');

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map((entry) => entry.isDirectory()
    ? files(resolve(directory, entry.name)) : [resolve(directory, entry.name)]))).flat();
}

export async function checkBuild() {
  const allFiles = await files(dist);
  const htmlFiles = allFiles.filter((file) => file.endsWith('.html'));
  const parsed = new Map();
  const errors = [];
  let checkedLinks = 0;
  for (const file of htmlFiles) parsed.set(file, load(await readFile(file, 'utf8')));
  for (const [file, $] of parsed) {
    const localPath = `${BASE_PATH}/${relative(dist, file).split(sep).join('/')}`;
    for (const element of $('[href], [src]').toArray()) {
      // GitHub Pages serves the error document as 404.html, not its canonical route.
      if (file === resolve(dist, '404.html') && $(element).is('link[rel="canonical"], link[rel="alternate"]')) continue;
      const value = $(element).attr('href') ?? $(element).attr('src');
      if (!value || /^(?:mailto:|tel:|data:|javascript:)/i.test(value)) continue;
      const url = new URL(value, SITE_ORIGIN + localPath);
      if (url.origin !== SITE_ORIGIN) continue;
      if (!url.pathname.startsWith(`${BASE_PATH}/`)) {
        errors.push(`${localPath}: URL drops the GitHub Pages base path: ${value}`);
        continue;
      }
      const path = decodeURIComponent(url.pathname.slice(BASE_PATH.length + 1));
      const target = resolve(dist, path.endsWith('/') || !path ? `${path}index.html` : path);
      const inside = relative(dist, target);
      if (inside === '..' || inside.startsWith(`..${sep}`)) {
        errors.push(`${localPath}: URL escapes build output: ${value}`);
        continue;
      }
      try { await access(target); }
      catch { errors.push(`${localPath}: missing target: ${value}`); continue; }
      if (url.hash && parsed.has(target)) {
        const id = decodeURIComponent(url.hash.slice(1));
        if (id && !parsed.get(target)('[id]').toArray().some((node) => parsed.get(target)(node).attr('id') === id)) {
          errors.push(`${localPath}: missing fragment: ${value}`);
        }
      }
      checkedLinks++;
    }
  }
  const slugs = ['', 'training', ...SOURCE_PAGES.map((page) => page.slug), ...README_PAGES.map((page) => page.slug)];
  if (!parsed.has(resolve(dist, 'index.html'))) errors.push('Default-language root redirect is missing.');
  for (const locale of LOCALES) {
    for (const slug of slugs) {
      const file = resolve(dist, locale, slug, 'index.html');
      const $ = parsed.get(file);
      if (!$) { errors.push(`Missing language route: ${route(locale, slug)}`); continue; }
      const expectedLanguage = locale === 'en' ? 'en' : 'zh-CN';
      if ($('html').attr('lang') !== expectedLanguage) errors.push(`Wrong document language: ${route(locale, slug)}`);
      const editLinks = $(`a[href^="${REPOSITORY}/edit/main/"]`);
      if (!slug && editLinks.length) errors.push(`Homepage should not display an edit link: ${locale}`);
      if (slug && !editLinks.length) errors.push(`Reading page source edit link is missing: ${route(locale, slug)}`);
      const brand = $('a.brand-title');
      if (brand.length !== 1 || brand.attr('aria-label') !== 'advai-notebooks' || brand.attr('href') !== route(locale, '')) {
        errors.push(`Brand link is missing, inaccessible or uses the wrong locale: ${route(locale, slug)}`);
      }
      if (brand.find('.brand-name').text() !== 'advai-notebooks' || brand.find('img.brand-mark[alt=""][aria-hidden="true"]').length !== 2) {
        errors.push(`Selected brand artwork or complete wordmark is missing: ${route(locale, slug)}`);
      }
      if (!$(`link[rel~="icon"][href="${BASE_PATH}/brand-mark.png"]`).length || !$(`link[rel~="icon"][href="${BASE_PATH}/brand-mark-dark.png"][media="(prefers-color-scheme: dark)"]`).length) {
        errors.push(`Theme-aware brand favicons are missing: ${route(locale, slug)}`);
      }
      const searchTranslations = JSON.parse($('site-search').attr('data-translations') ?? '{}');
      if (locale === 'zh-cn' && !searchTranslations.many_results?.includes('找到')) {
        errors.push(`Chinese search messages are missing: ${route(locale, slug)}`);
      }
      const values = $('starlight-lang-select option').toArray().map((node) => $(node).attr('value'));
      const other = locale === 'en' ? 'zh-cn' : 'en';
      if (!values.includes(route(other, slug))) errors.push(`Language switch does not retain page: ${route(locale, slug)}`);
    }
    const overview = parsed.get(resolve(dist, locale, 'training/index.html'));
    const home = parsed.get(resolve(dist, locale, 'index.html'));
    const facts = home?.('.landing-facts dt').toArray().map((node) => home(node).text().replace(/\s+/g, ' ').trim());
    const expectedFacts = [TRACKS.length.toString().padStart(2, '0'), notebookCount.toString().padStart(2, '0'), 'Notebook + CLI'];
    if (JSON.stringify(facts) !== JSON.stringify(expectedFacts)) errors.push(`Homepage overview must show track/notebook counts and the shared workflow: ${locale}`);
    const workflowLabel = locale === 'en' ? 'Shared training code' : '同一套训练实现';
    if (home?.('.landing-facts dd').eq(2).text() !== workflowLabel) errors.push(`Homepage workflow label is not localized: ${locale}`);
    if (home?.('.training-tile').length !== TRACKS.length) errors.push(`Homepage training directory is incomplete: ${locale}`);
    if (home?.('.training-filters button[aria-pressed="true"]').length !== 1) errors.push(`Homepage filters have no unique initial selection: ${locale}`);
    if (!home?.('.landing-visual img[src$=".webp"]').length) errors.push(`Optimized homepage illustration is missing: ${locale}`);
    for (const track of TRACKS) {
      const notebook = notebookPath(track, locale);
      if (notebook && !overview?.(`a[href="${colabUrl(notebook)}"]`).length) errors.push(`Missing localized Colab entry: ${locale}/${track.name}`);
      if (notebook && !home?.(`a[href="${colabUrl(notebook)}"]`).length) errors.push(`Missing homepage Colab entry: ${locale}/${track.name}`);
    }
  }
  const searchFiles = allFiles.filter((file) => relative(dist, file).startsWith(`pagefind${sep}`));
  if (!searchFiles.some((file) => file.endsWith('pagefind.js')) || !searchFiles.some((file) => file.endsWith('.pf_meta'))) {
    errors.push('Production search index is missing.');
  }
  for (const page of SOURCE_PAGES.filter((page) => !page.sources['zh-cn'])) {
    const $ = parsed.get(resolve(dist, 'zh-cn', page.slug, 'index.html'));
    if (!$?.('section[lang="en"]').length || !$?.('aside').text().includes('暂未提供中文正文')) {
      errors.push(`Untranslated guide is not identified: ${page.slug}`);
    }
  }
  if (errors.length) throw new Error(`Build validation failed:\n${errors.join('\n')}`);
  console.log(`Validated ${htmlFiles.length} HTML files, ${checkedLinks} local links/assets, matching bilingual switches, localized Colab links and search index.`);
}

await checkBuild();
