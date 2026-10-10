export const REPOSITORY = 'https://github.com/advai-x/advai-notebooks';
export const SITE_ORIGIN = 'https://advai-x.github.io';
export const BASE_PATH = '/advai-notebooks';
export const LOCALES = ['en', 'zh-cn'];

export function route(locale, slug = '') {
  if (!LOCALES.includes(locale)) throw new Error(`Unsupported locale: ${locale}`);
  return `${BASE_PATH}/${locale}/${slug ? `${slug}/` : ''}`;
}

const page = (slug, english, chinese, en, zh) => ({
  slug,
  title: { en: english, 'zh-cn': chinese },
  sources: { en, ...(zh ? { 'zh-cn': zh } : {}) },
});

// Only these reader-facing sources are published. Other repository links stay on GitHub.
export const SOURCE_PAGES = [
  page('guides/embedding', 'Embedding fine-tuning', 'Embedding 微调', 'notebooks/embedding/README.md'),
  page('guides/ranker', 'Ranker fine-tuning', 'Ranker 微调', 'notebooks/ranker/README.md'),
  page('guides/decision', 'Decision-model fine-tuning', 'Decision 模型微调', 'notebooks/decision/README.md'),
  page('guides/llm', 'LLM Base & Instruct', 'LLM Base 与 Instruct', 'docs/llm.md', 'docs/llm.zh-CN.md'),
  page('guides/rl', 'LLM PPO & GRPO', 'LLM PPO 与 GRPO', 'docs/rl.md', 'docs/rl.zh-CN.md'),
  page('guides/diffusion', 'Diffusion', 'Diffusion', 'docs/diffusion.md', 'docs/diffusion.zh-CN.md'),
  page('guides/stt', 'STT: speech-to-text', 'STT：语音转文字', 'docs/stt.md', 'docs/stt.zh-CN.md'),
  page('guides/tts', 'TTS: text-to-speech', 'TTS：文字转语音', 'docs/tts.md', 'docs/tts.zh-CN.md'),
  page('guides/vlm', 'VLM: image question answering', 'VLM：图像问答', 'docs/vlm.md', 'docs/vlm.zh-CN.md'),
  page('guides/vla', 'VLA: action-training baseline', 'VLA：动作训练基线', 'docs/vla.md', 'docs/vla.zh-CN.md'),
  page('guides/vm', 'VM: World Model', 'VM：World Model', 'docs/vm.md', 'docs/vm.zh-CN.md'),
  page('reference/architecture', 'Architecture & extension boundaries', '架构与扩展边界', 'docs/architecture.md', 'docs/architecture.zh-CN.md'),
  page('reference/datasets', 'Datasets & recorded demo releases', '数据与已记录的 Demo 发布', 'docs/datasets.md', 'docs/datasets.zh-CN.md'),
  page('reference/publishing', 'Publish to Hugging Face', '发布到 Hugging Face', 'docs/publishing.md', 'docs/publishing.zh-CN.md'),
  page('reference/retrieval-data', 'Retrieval data contract', '检索数据契约', 'data/sample/retrieval/README.md'),
  page('development/docs-site', 'Maintain this documentation site', '维护文档站', 'site/README.md', 'site/README.zh-CN.md'),
];

export const README_PAGES = [
  {
    slug: 'start/quickstart',
    title: { en: 'Quick start', 'zh-cn': '快速开始' },
    sections: { en: ['Quick start'], 'zh-cn': ['快速开始'] },
    intro: {
      en: 'Choose a guide from the [training directory](notebooks/README.md), then start with its tiny smoke configuration. The Colab instructions below use Embedding as an example.',
      'zh-cn': '从[训练目录](notebooks/README.zh-CN.md)选择指南，先使用 tiny smoke 配置。下方 Colab 安装命令以 Embedding 为例。',
    },
  },
  {
    slug: 'start/workflow',
    title: { en: 'From smoke run to real experiment', 'zh-cn': '从 smoke 到真实实验' },
    sections: {
      en: ['From smoke test to a real experiment', 'Data, secrets and generated files'],
      'zh-cn': ['从 smoke 到真实实验', '数据、凭据与生成产物'],
    },
  },
  {
    slug: 'reference/cli',
    title: { en: 'CLI & optional workflows', 'zh-cn': 'CLI 与可选流程' },
    sections: { en: ['Command-line overview'], 'zh-cn': ['CLI 与可选流程'] },
  },
  {
    slug: 'development/contributing',
    title: { en: 'Development & contributing', 'zh-cn': '开发与贡献' },
    sections: {
      en: ['Development', 'Contributing', 'Project status', 'License'],
      'zh-cn': ['开发与贡献', '状态与许可证'],
    },
  },
];

// Descriptions and entry points only; detailed guide prose remains in the sources above.
export const TRACKS = [
  { name: 'Embedding', guide: 'guides/embedding', extra: 'embedding', config: 'configs/embedding/smoke.yaml', notebook: 'notebooks/embedding/01_finetune.ipynb', description: { en: 'Dense text retrieval with MNRL and hard-negative preparation.', 'zh-cn': '使用 MNRL 与困难负例准备进行文本稠密检索训练。' } },
  { name: 'Ranker', guide: 'guides/ranker', extra: 'ranker', config: 'configs/ranker/smoke.yaml', notebook: 'notebooks/ranker/01_finetune.ipynb', description: { en: 'CrossEncoder reranking with pointwise BCE and ranking evaluation.', 'zh-cn': '使用 Pointwise BCE 与排序评估训练 CrossEncoder 重排模型。' } },
  { name: 'Decision', guide: 'guides/decision', extra: 'decision', config: 'configs/decision/smoke.yaml', notebook: 'notebooks/decision/01_finetune.ipynb', description: { en: 'Typed choice, score and yes/no decisions with optional calibration.', 'zh-cn': '训练 choice、score 与 yes/no 类型化决策，并支持可选校准。' } },
  { name: 'LLM Base', guide: 'guides/llm', extra: 'llm-base', config: 'configs/llm/base/smoke.yaml', notebook: 'notebooks/llm/base/01_finetune.ipynb', description: { en: 'Continued pre-training and full or LoRA decoder fine-tuning.', 'zh-cn': 'Decoder 继续预训练，以及 full 或 LoRA 微调。' } },
  { name: 'LLM Instruct', guide: 'guides/llm', extra: 'llm-instruct', config: 'configs/llm/instruct/smoke.yaml', notebook: 'notebooks/llm/instruct/01_finetune.ipynb', description: { en: 'Assistant-only supervised fine-tuning for instruction-following models.', 'zh-cn': '面向指令模型的 Assistant-only 监督微调。' } },
  { name: 'LLM PPO', guide: 'guides/rl', extra: 'llm-ppo', config: 'configs/llm/ppo/smoke.yaml', description: { en: 'Online policy optimization with a value head, GAE and clipping. CLI guide only.', 'zh-cn': '使用价值头、GAE 与 clipping 进行在线策略优化；当前提供 CLI 指南。' } },
  { name: 'LLM GRPO', guide: 'guides/rl', extra: 'llm-grpo', config: 'configs/llm/grpo/smoke.yaml', description: { en: 'Group-relative policy optimization without a critic. CLI guide only.', 'zh-cn': '无需 critic 的组内相对策略优化；当前提供 CLI 指南。' } },
  { name: 'Diffusion', guide: 'guides/diffusion', extra: 'diffusion', config: 'configs/diffusion/smoke.yaml', notebook: 'notebooks/diffusion/01_finetune.ipynb', description: { en: 'Latent text-to-image training with full UNet updates or attention LoRA.', 'zh-cn': '支持 Full UNet 更新或 attention LoRA 的 latent 文生图训练。' } },
  { name: 'VLM', guide: 'guides/vlm', extra: 'vlm', config: 'configs/vlm/smoke.yaml', notebook: 'notebooks/vlm/01_finetune.ipynb', description: { en: 'Native LLaVA single-image question answering, full or language LoRA.', 'zh-cn': '原生 LLaVA 单图问答，支持 full 或语言 LoRA。' } },
  { name: 'VLA', guide: 'guides/vla', extra: 'vla', config: 'configs/vla/smoke.yaml', notebook: 'notebooks/vla/01_finetune.ipynb', description: { en: 'CLIP, language and state to continuous action chunks: a behavior-cloning baseline.', 'zh-cn': 'CLIP、语言与状态到连续动作 chunk 的行为克隆基线。' } },
  { name: 'VM / World Model', guide: 'guides/vm', extra: 'vm', config: 'configs/vm/smoke.yaml', notebook: 'notebooks/vm/01_finetune.ipynb', description: { en: 'Action-conditioned numerical dynamics and recursive rollouts, not video generation.', 'zh-cn': '动作条件数值动力学与递归 rollout，不是视频生成模型。' } },
  { name: 'STT', guide: 'guides/stt', extra: 'stt', config: 'configs/stt/smoke.yaml', notebook: 'notebooks/stt/01_finetune.ipynb', description: { en: 'Native Whisper supervised transcription, WER/CER and epoch recovery.', 'zh-cn': '原生 Whisper 监督转写、WER/CER 与 epoch 恢复。' } },
  { name: 'TTS', guide: 'guides/tts', extra: 'tts', config: 'configs/tts/smoke.yaml', notebook: 'notebooks/tts/01_finetune.ipynb', description: { en: 'Qwen3-TTS Base single-speaker fine-tuning and non-streaming synthesis; synthetic offline smoke.', 'zh-cn': 'Qwen3-TTS Base 单说话人微调与非流式生成；提供合成离线 smoke。' } },
];

export function notebookPath(track, locale) {
  return track.notebook && (locale === 'zh-cn' ? track.notebook.replace(/\.ipynb$/, '.zh-CN.ipynb') : track.notebook);
}

export function repositoryUrl(file, directory = false) {
  return `${REPOSITORY}/${directory ? 'tree' : 'blob'}/main/${file.split('/').map(encodeURIComponent).join('/')}`;
}

export function colabUrl(file) {
  return `https://colab.research.google.com/github/advai-x/advai-notebooks/blob/main/${file}`;
}
