# 文档站

[English](README.md) | 简体中文

文档站使用 Astro Starlight，将现有训练指南发布到 GitHub Pages。
英文为默认语言，简体中文使用对应页面路径。预计发布入口为
[英文版](https://advai-x.github.io/advai-notebooks/en/)和
[中文版](https://advai-x.github.io/advai-notebooks/zh-cn/)。推送工作流并启用 Pages 后，这些地址才会可用。

## 本地开发

需要 Node.js 22.12 或更新版本，以及 npm；不需要 Python ML 依赖或 GPU。
从仓库根目录运行：

```bash
cd site
npm ci
npm run check
npm run dev
```

开发服务器会输出本地地址，通常为 `http://127.0.0.1:4321/advai-notebooks/`。
根路径会跳转到英文版，中文入口为 `/advai-notebooks/zh-cn/`。

服务器运行期间修改原始指南后，在另一个终端执行 `npm run sync-docs` 更新生成页面。
站点配置与样式修改由 Astro 开发服务器自动处理。

## 构建与生产预览

```bash
cd site
npm ci
npm run check
npm run build
npm run preview
```

搜索索引在生产构建时生成，因此请使用生产预览测试搜索。
构建会检查站内链接、锚点、静态资源、双语路径和搜索索引；不会抓取外部服务。
Notebook 入口通过仓库源文件校验，不会下载或执行 Notebook。
输出目录为 `site/dist/`，已被 Git 忽略。

Astro 可能在后台启动预览服务。预览结束后，在 `site/` 执行
`npm exec -- astro preview stop` 停止服务，再重新安装依赖。

## 保持唯一内容来源

- 详细指南正文仍在原有 `docs/`、`notebooks/` 或数据契约 Markdown 中维护。
- [scripts/catalog.mjs](scripts/catalog.mjs) 定义发布白名单、导航元数据与 Notebook / 配置入口。
- [scripts/sync-content.mjs](scripts/sync-content.mjs) 生成首页与训练目录元数据，并从根 README 的对应版本提取快速开始、实验流程、CLI 与贡献章节，不手工复制这些正文。
- `src/components/` 中的首页组件展示训练目录、分类筛选与原始 README smoke 示例。[scripts/landing.mjs](scripts/landing.mjs) 维护共享分组元数据，原有文档 URL 保持不变。
- 转换前先解析 Markdown；代码块和行内代码不会被当成链接或修改为另一份源码。
- 已发布的指南跳转到当前语言的站内页面；其他文件和目录链接到 GitHub；Notebook 按钮另提供 Colab 入口。
- 原指南标题和语言切换行由 Starlight 的标题与语言选择器替代，GitHub 提醒块转换为站点提示框。
- 缺少中文源文件时，中文页面会显式标明英文回退，标注正文语言，并在已有中文 Notebook 时提供对应入口。翻译完成后，在文档目录中登记中文文件。
- 编辑入口始终指向原始文件，而不是生成副本。

`site/src/content/docs/`、`site/public/generated/` 与 `site/.content-manifest.json` 均为被忽略的生成产物，
不要手工修改或提交。Astro 将 `src/assets/` 中的首页插图优化为响应式 WebP，Geist 字体也在本地托管。
README 原头图保持不变，不发布本地数据、凭据、权重或训练输出。
`BrandTitle.astro` 在首页与文档页统一展示选中的 Notebook 图形和可选择的文字字标。
深浅主题素材及生成记录见 [src/assets/brand.zh-CN.md](src/assets/brand.zh-CN.md)。

## 启用 GitHub Pages

仓库管理员需完成一次设置：

1. 在仓库中打开 **Settings → Pages**。
2. 将 **Build and deployment → Source** 设置为 **GitHub Actions**。
3. 将站点和工作流推送到 `main`，或者启用后手动运行 **Documentation** 工作流。
4. 检查部署任务与 `github-pages` 环境中的站点地址。

[文档工作流](../.github/workflows/docs.yml) 在 PR 中执行测试与构建，只有 `main` 的 push 或手动运行才部署；
PR 不具备 Pages 写权限。发布产物仅包含 `site/dist/`，不会打包整个仓库或训练产物。

工作流只安装 Node 依赖，不导入 Python 包、不下载模型权重、不运行训练单元格。
提交 lockfile，使用 `npm ci` 保持可复现安装。首版不配置自定义域名或文档版本切换。

`postcss-selector-parser` 的依赖 override 用于避开代码高亮 CSS 依赖链中的
[GHSA-rj75-hqrm-r3gf](https://github.com/advisories/GHSA-rj75-hqrm-r3gf)。
上游更新依赖后应重新评估此覆盖配置；调整时保留完整生产构建验证。

## 增加或更新指南

1. 更新原始英文指南与已有的中文版本。
2. 在 `scripts/catalog.mjs` 中登记源文件和入口。
3. 在 `astro.config.mjs` 中加入对应侧栏分组。
4. 执行 `npm run check` 与 `npm run build`，再检查两个语言路径。
5. 新增模型路径时，确认 Colab 链接、配置路径和能力边界提示。

参考 [Starlight 官方文档](https://starlight.astro.build/)与
[GitHub Pages 部署指南](https://docs.astro.build/en/guides/deploy/github/)。
