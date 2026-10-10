# Documentation site

English | [简体中文](README.zh-CN.md)

The Astro Starlight site publishes the existing training guides on GitHub Pages.
English is the default; Simplified Chinese uses matching routes. The planned
production entry points are [English](https://advai-x.github.io/advai-notebooks/en/)
and [简体中文](https://advai-x.github.io/advai-notebooks/zh-cn/). These URLs become
available after the workflow is pushed and GitHub Pages is enabled.

## Local development

Use Node.js 22.12 or newer and npm. No Python ML packages or GPU are needed.
From the repository root:

```bash
cd site
npm ci
npm run check
npm run dev
```

The development server prints its local address, normally
`http://127.0.0.1:4321/advai-notebooks/`. The root redirects to English.
Navigate to `/advai-notebooks/zh-cn/` for Chinese.

After editing a source guide while the server is running, run `npm run sync-docs`
in another terminal to refresh generated content. Editing the site configuration
or styles is handled by Astro's development server.

## Production build and preview

```bash
cd site
npm ci
npm run check
npm run build
npm run preview
```

Search is indexed during the production build, so use the production preview
to test it. The build checks internal links, fragment anchors, assets, bilingual
routes and the presence of a search index. External services are not crawled.
Notebook links are checked against local source files rather than downloading
or executing notebooks. Build output is in `site/dist/` and is ignored by Git.

Astro may start preview as a background server. When finished, stop the preview
with `npm exec -- astro preview stop` from `site/` before reinstalling dependencies.

## One source of truth

- Edit detailed guide prose in the original `docs/`, `notebooks/` or data-contract
  Markdown files, not in generated site pages.
- [scripts/catalog.mjs](scripts/catalog.mjs) selects published sources, navigation
  metadata and verified Notebook/configuration entry points.
- [scripts/sync-content.mjs](scripts/sync-content.mjs) generates home and training
  directory metadata and extracts quick-start, workflow, CLI and contributor sections
  from the root README translations. It does not duplicate those sections manually.
- Homepage components in `src/components/` render the training catalog, filters
  and original README smoke example. Shared group metadata is in
  [scripts/landing.mjs](scripts/landing.mjs). The documentation URLs stay unchanged.
- Markdown is parsed before transforming links. Fenced code and inline code
  examples are not interpreted as links or modified as source code.
- Published documents link to matching-language site routes. Unpublished files
  and directories link to GitHub; Notebook buttons also offer Colab.
- Source headings and language-switch lines are replaced by Starlight's title
  and language selector. GitHub alerts are rendered as Starlight asides.
- If a guide has no Chinese source, the Chinese page explicitly identifies the
  English fallback, marks its body language and offers Chinese Notebook links
  where available. Register a translation in the catalog when it is ready.
- Edit links always point to the original source file, not the generated copy.

Generated files under `site/src/content/docs/`, `site/public/generated/` and
`site/.content-manifest.json` are ignored. Do not hand-edit or commit them.
The homepage illustration in `src/assets/` is optimized to responsive WebP assets
by Astro. Geist fonts are served locally. The original README banner is unchanged;
local datasets, credentials, checkpoints and training outputs are not deployed.
The shared `BrandTitle.astro` component renders the selected notebook mark and a
live-text wordmark on both home and reading pages. Light/dark assets and their
generation provenance are documented in [src/assets/brand.md](src/assets/brand.md).

## Enable GitHub Pages

An administrator must perform this one-time repository setting:

1. Open **Settings → Pages** in the GitHub repository.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Push this site's files and its workflow to `main`, or run the **Documentation**
   workflow manually after the setting is enabled.
4. Check the workflow's deploy job and the `github-pages` environment URL.

The [documentation workflow](../.github/workflows/docs.yml) tests and builds on
pull requests. Deployment only runs for `main` pushes or manual runs on `main`;
pull requests do not get Pages write permissions. The deployment artifact contains
only `site/dist/`, not the repository or training artifacts.

The workflow installs only Node dependencies and never imports the Python package,
downloads model weights or runs training cells. The lockfile is committed for
repeatable `npm ci` installs. No custom domain or documentation-version selector
is configured in this first version.

The `postcss-selector-parser` dependency override avoids advisory
[GHSA-rj75-hqrm-r3gf](https://github.com/advisories/GHSA-rj75-hqrm-r3gf) in the
code-highlighting CSS dependency chain. Re-evaluate it when the upstream
dependency is updated; keep the production build test when changing it.

## Add or update a guide

1. Update its original English guide and Chinese companion when available.
2. Add source mappings and entry points in `scripts/catalog.mjs`.
3. Add its slug to the appropriate sidebar group in `astro.config.mjs`.
4. Run `npm run check` and `npm run build`, then inspect both language routes.
5. For a new model family, check Colab links, configuration paths and scope warnings.

See the official [Starlight documentation](https://starlight.astro.build/) and
[GitHub Pages deployment guide](https://docs.astro.build/en/guides/deploy/github/).
