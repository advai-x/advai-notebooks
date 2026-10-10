import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

async function pngInfo(path) {
  const bytes = await readFile(new URL(path, import.meta.url));
  assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20), colorType: bytes[25] };
}

test('selected theme marks have matching square RGBA canvases', async () => {
  const light = await pngInfo('../src/assets/notebook-mark-light.png');
  const dark = await pngInfo('../src/assets/notebook-mark-dark.png');
  assert.deepEqual(light, dark);
  assert.equal(light.width, light.height);
  assert.ok(light.width >= 80);
  assert.equal(light.colorType, 6, 'Both original marks must retain an alpha channel.');
});

test('both theme favicons are compact 64px RGBA PNGs', async () => {
  for (const path of ['../public/brand-mark.png', '../public/brand-mark-dark.png']) {
    assert.deepEqual(await pngInfo(path), { width: 64, height: 64, colorType: 6 });
  }
});
