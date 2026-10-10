# Notebook brand mark

English | [简体中文](brand.zh-CN.md)

The site uses the second displayed logo concept selected for advai-notebooks:
two folded notebook pages with an N-shaped negative space. The generated raster
mark is used directly; it is not recreated with CSS or a hand-drawn SVG.

- `notebook-mark-light.png`: blue/navy source mark for light surfaces.
- `notebook-mark-dark.png`: color adaptation for dark surfaces, with matching geometry.
- `../../public/brand-mark.png` and `brand-mark-dark.png`: 64px PNG favicons,
  resized from the matching source with Sharp, without drawing or changing the design.

`BrandTitle.astro` uses Astro's optimized 80px WebP variants for high-density
displays. The full project name remains live UI text in the locally served Geist
Mono font, retaining its semantic label, selectable text and crisp small-size
rendering. The mark is decorative because the home link already has its full
accessible name. Both locales retain their own home-link destinations.

## Image-generation provenance

The built-in image-generation tool extracted the selected design and produced
the theme adaptation. The following are the final source-asset prompts.

### Light mark

Use case: background-extraction.
Asset type: standalone transparent website logo mark.
Input image 1 is the selected open-notebook logo design. Extract ONLY the
two-page notebook symbol with the negative-space letter N from the large logo
at the top. Remove all lettering, wordmark, webpage, divider, background and
other objects.
Preserve the exact source symbol: blue left page, navy right page, central
N-shaped negative space, two broad folded page shapes, original silhouette,
proportions, corner treatment and subtle original color shading. Do not simplify
or redraw it as a generic book, mountain, A, cube, app badge or different monogram.
The N cutout must be genuinely transparent.
Output one centered faithful symbol only, on a genuinely transparent square
canvas. The symbol should fill about 88% of the square height with equal
breathing space all around, suitable for a 28px navigation mark. Clean crisp
edges. No letters outside the mark, no shadow, no opaque white rectangle, no
painted checkerboard, no border, no labels, no watermark.

### Dark mark

Use case: precise-object-edit.
Asset type: transparent dark-theme adaptation of the supplied website brand mark.
Input image 1 is the exact selected open two-page notebook symbol with a
negative-space N. Change ONLY its foreground colors for visibility on a dark
#101720 website. Preserve the canvas size, symbol position, exact geometry,
silhouette, page folds, N cutout, padding and transparent areas. Do not redraw
or create a new logo.
Left page: restrained pale blue #a8d8f7. Right page: muted lighter steel blue
#7792b1. Preserve the subtle source shading without adding any new lighting
effect. Keep all negative space genuinely transparent and clean. No text,
extra shapes, labels, rectangle, shadow, glow, background or watermark.
One symbol only, identical geometry and alignment to the source.
