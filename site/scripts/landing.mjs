import { TRACKS } from './catalog.mjs';

export const TRAINING_GROUPS = [
  { id: 'retrieval', extras: ['embedding', 'ranker', 'decision'], label: { en: 'Retrieval & decisions', 'zh-cn': '检索与决策' } },
  { id: 'language', extras: ['llm-base', 'llm-instruct', 'llm-ppo', 'llm-grpo'], label: { en: 'Language models', 'zh-cn': '语言模型' } },
  { id: 'multimodal', extras: ['diffusion', 'vlm', 'vla', 'vm'], label: { en: 'Multimodal & generation', 'zh-cn': '多模态与生成' } },
  { id: 'speech', extras: ['vad', 'stt', 'tts'], label: { en: 'Speech models', 'zh-cn': '语音模型' } },
];

export function familyFor(track) {
  const group = TRAINING_GROUPS.find(({ extras }) => extras.includes(track.extra));
  if (!group) throw new Error(`Training track has no homepage group: ${track.extra}`);
  return group.id;
}

export function isHomepage(route) {
  return ['en', 'zh-cn'].includes(route.locale) && route.id.replace(/\/index$/, '') === route.locale;
}

export const notebookCount = TRACKS.filter((track) => track.notebook).length;
