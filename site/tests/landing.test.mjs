import assert from 'node:assert/strict';
import test from 'node:test';
import { TRACKS } from '../scripts/catalog.mjs';
import { TRAINING_GROUPS, familyFor, isHomepage, notebookCount } from '../scripts/landing.mjs';

test('every training track belongs to exactly one homepage family', () => {
  for (const track of TRACKS) {
    const matching = TRAINING_GROUPS.filter((group) => group.extras.includes(track.extra));
    assert.equal(matching.length, 1, track.name);
    assert.equal(familyFor(track), matching[0].id);
  }
  assert.equal(new Set(TRAINING_GROUPS.flatMap((group) => group.extras)).size, TRACKS.length);
  assert.throws(() => familyFor({ extra: 'unknown-model' }), /no homepage group/);
});

test('homepage overrides never replace reading pages or error pages', () => {
  assert.ok(isHomepage({ id: 'en', locale: 'en' }));
  assert.ok(isHomepage({ id: 'zh-cn/index', locale: 'zh-cn' }));
  assert.ok(!isHomepage({ id: 'en/guides/vlm', locale: 'en' }));
  assert.ok(!isHomepage({ id: '404', locale: 'en' }));
  assert.ok(!isHomepage({ id: 'en/404', locale: 'en' }));
});

test('homepage notebook counts describe primary guides, excluding the optional workflow', () => {
  assert.equal(notebookCount, 11);
  assert.equal(TRACKS.length, 13);
  assert.equal(familyFor(TRACKS.find((track) => track.extra === 'stt')), 'speech');
  assert.equal(familyFor(TRACKS.find((track) => track.extra === 'tts')), 'speech');
  assert.equal(TRACKS.length - notebookCount, 2);
});
