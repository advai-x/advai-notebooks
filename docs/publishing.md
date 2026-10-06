# 将模型发布到 Hugging Face

完整链路现在为：数据 → 训练/微调 → 评估/选模 → 原生导出 → 可选发布 → 按版本加载推理。
Embedding、Ranker、Decision、LLM Base、LLM Instruct 各自发布到独立 model 仓库，使用相同命令参数。
发布是显式动作，`run` 不会自动上传。五个微调 Notebook 的最后一节也提供发布入口，默认只做离线预览。

## 安装和认证

```bash
python -m pip install -e '.[embedding,ranker,decision,publish]'
hf auth login
```

也可以通过环境变量 `HF_TOKEN` 使用具备目标仓库写权限的凭据。
Token 不放进 YAML、模型卡、发布清单或命令行参数。Python API 的 `token` 仅传给 Hub 客户端。

## 先验证发布包

`--export-dir` 指向推理导出目录，例如 `run` 生成的 `output_dir/export`。
程序会筛选推理文件、生成模型卡，再从筛选后的发布包实际加载模型；缺少必需文件会在上传前失败。
离线预览不访问 Hub，也不读取登录凭据。

```bash
python -m advai_notebooks.embedding publish \
  --export-dir outputs/embedding/smoke/export \
  --repo-id your-org/your-embedding --dry-run \
  --output outputs/publishing/embedding-plan.json

python -m advai_notebooks.ranker publish \
  --export-dir outputs/ranker/smoke/export \
  --repo-id your-org/your-ranker --dry-run \
  --output outputs/publishing/ranker-plan.json

python -m advai_notebooks.decision publish \
  --export-dir outputs/decision/smoke/export \
  --repo-id your-org/your-decision --dry-run \
  --output outputs/publishing/decision-plan.json
```

预览结果包括仓库、可见性、目标分支、文件大小/SHA256 和完整模型卡。
它不会改动原始导出。只有原始导出模型文件会被筛选上传，不上传运行目录、原始数据、训练标签、
candidate 文件、优化器/随机状态、运行日志或 `.env`。

| 任务 | 发布内容 | 加载/推理 |
|---|---|---|
| Embedding | SentenceTransformer 原生组件、权重、tokenizer、prefix 元数据 | `EmbeddingModel.load/encode`，或原生 SentenceTransformer |
| Ranker | CrossEncoder 原生组件、权重、tokenizer、prefix/原始分数元数据 | `RankerModel.load/score`，或原生 CrossEncoder |
| Decision | 完整 backbone/head 权重、encoder 配置、tokenizer、任务定义和校准参数 | `DecisionModel.load/system_one` |
| LLM Base | full/adapter 权重、tokenizer、generation config、基座来源 | `BaseModel.load/generate` |
| LLM Instruct | full/adapter 权重、tokenizer/chat template、generation config、基座来源 | `InstructModel.load/chat` |

任务定义中的问题、选项和规则属于 Decision 推理契约，`tasks.json` 会随模型发布。
生成的 `README.md` 提供模型卡、推理示例及随机 tiny 模型说明；如果导出旁有匹配的运行报告，
还会包含选中的 epoch 和测试集汇总指标。完整实验报告和样例预测不上传。
发布用的标准模型元数据会去除认证字段，并将绝对的本地模型路径替换为 `local`；权重和任务定义不改动。

## 实际发布

确认使用自己的仓库名和经过领域验证的模型后，去掉 `--dry-run`：

```bash
python -m advai_notebooks.decision publish \
  --export-dir outputs/decision/domain-run/export \
  --repo-id your-org/your-decision \
  --model-card docs/model-cards/your-decision.md \
  --output outputs/publishing/decision-result.json
```

Embedding、Ranker 使用完全相同的参数。新仓库默认私有；加 `--public` 发布到公开仓库。
对于已有仓库，所选可见性必须与已有设置一致；发布命令不会自动修改已有仓库的可见性。
每个仓库存放一种模型。更新仓库会添加/覆盖本次发布文件，不删除远端其他文件。

`--revision` 默认为 `main`，其他值需要对应已存在的可写分支。
可以通过 `--license` 设置生成模型卡的 license 标识；默认不推断许可证。
通过 `--model-card` 提供的自定义 Markdown 会原样使用，包括其中的许可证和训练/评估说明。
自定义卡优先于自动生成的卡。

发布到 Hub 的模型卡使用英文。当前 demo 的英文源文件保存在
[`docs/model-cards/wbs-embedding-demo.md`](model-cards/wbs-embedding-demo.md)、
[`docs/model-cards/wbs-ranker-demo.md`](model-cards/wbs-ranker-demo.md)、
[`docs/model-cards/wbs-decision-demo.md`](model-cards/wbs-decision-demo.md)，
可以通过 `--model-card` 显式使用。模型的输入语言和 tokenizer 内容仍由模型本身决定。

发布包包含 `publish_manifest.json`，记录 schema 版本、任务身份及每个推理文件的 SHA256。
成功结果包含仓库 URL、commit SHA 和 commit URL；保留此结果便于部署时固定版本。
网络或权限错误会直接返回失败，不会记成发布成功。

## 按版本加载推理

三个 `Model.load` 均接受本地导出路径或本项目发布的 `namespace/model`，以及 `revision`、`token`、`cache_dir`、
`local_files_only`。Hub 通过 snapshot 下载完整模型，之后走已有原生加载器。

```python
from advai_notebooks.embedding.model import EmbeddingModel
from advai_notebooks.ranker.model import RankerModel
from advai_notebooks.decision.model import DecisionModel

embedding = EmbeddingModel.load("your-org/your-embedding", revision="EMBEDDING_COMMIT_SHA")
vectors = embedding.encode(["reset password"], role="query")

ranker = RankerModel.load("your-org/your-ranker", revision="RANKER_COMMIT_SHA")
scores = ranker.score([("reset password", "account login password reset")])

decision = DecisionModel.load("your-org/your-decision", revision="DECISION_COMMIT_SHA")
answers = decision.system_one("Customer message", task="your-task-id")
```

CLI 的 `encode` / `rerank` / `predict` / `evaluate` 也支持 `--revision`、`--cache-dir`、
`--local-files-only`。私有仓库加载使用登录凭据或 `HF_TOKEN`。
Decision 是本项目自有架构，需安装本项目后使用其加载器；普通 Transformers `AutoModel`
不会重建决策 head、输入编码或类型化解码。上传模型文件也不会自动部署托管推理服务。

## 实现与验证

各任务 `publish.py` 决定自己的文件契约和模型卡，`common/hub.py` 只处理目录筛选、打包、
Hub 上传和 snapshot 下载，不导入其他任务或任何指标实现。
Token/模型卡/仓库设置不加入训练 YAML，训练依赖独立性继续保持。

验证覆盖三个真实 tiny 原生导出的打包、离线 CLI、模拟上传、模拟 Hub 下载重载、
prefix/分数/三种决策输出一致、任务与校准参数保留、文件哈希、路径检查和已有仓库可见性检查。
离线测试不会创建真实 Hub 仓库；真实上传需要用户提供目标仓库和登录凭据。

官方参考：[文件上传](https://huggingface.co/docs/huggingface_hub/guides/upload)、
[模型卡](https://huggingface.co/docs/huggingface_hub/guides/model-cards)、
[snapshot 下载](https://huggingface.co/docs/huggingface_hub/guides/download)。


## LLM full/adapter 发布

Base/Instruct 使用相同 `publish` 参数。完整模型可直接原生加载，adapter 必须加载固定基座；
bundled base 会随包上传，外部基座记录不可变 SHA。Chat template `.jinja` 及多模板目录一并筛选。
离线预览校验 adapter 时仅使用缓存，缺少远程基座缓存会失败而不会下载。
使用 `export --merged` 生成不依赖 adapter 的完整模型；详见 [LLM guide](llm.md)。

```bash
python -m advai_notebooks.llm.base publish \
  --export-dir outputs/llm/base/run/export \
  --repo-id widebluesky/wbs-llm-base-demo --dry-run
python -m advai_notebooks.llm.instruct publish \
  --export-dir outputs/llm/instruct/run/export \
  --repo-id widebluesky/wbs-llm-instruct-demo --dry-run
```

这两个命名是后续 demo 发布目标，本次补齐框架没有创建或上传 LLM 仓库。
