# Notebook 品牌图形

[English](brand.md) | 简体中文

本站采用用户选中的第二张 Logo 设计：两页折叠 Notebook 形成 N 形负空间。
直接使用生成的图形素材，不用 CSS 或手绘 SVG 重新近似绘制。

- `notebook-mark-light.png`：浅色背景使用的蓝色 / 深蓝色图形。
- `notebook-mark-dark.png`：相同形状的深色主题配色版本。
- `../../public/brand-mark.png` 与 `brand-mark-dark.png`：使用 Sharp 等比缩小
  为 64px 的 PNG favicon，不绘图或修改设计内容。

`BrandTitle.astro` 使用 Astro 优化的 80px WebP，适配高像素密度屏幕。
完整项目名仍以本地 Geist Mono 字体显示为可选择的 UI 文字，保持小尺寸清晰。
图形作为装饰内容，首页链接已有完整的无障碍名称，且保留当前语言的首页地址。

## 图像生成记录

使用内置图像生成工具从选中的设计稿提取图形，再生成深色主题版本。
完整的原始英文提示词见 [English](brand.md#image-generation-provenance)。

浅色版本要求保留原始轮廓、折页、N 形透明负空间、比例和配色，移除其他内容；
深色版本只调整前景配色，不改变图形结构、位置和透明背景。
