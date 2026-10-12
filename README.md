# Jev-WAF-Lite：基于 BERT 的 HTTP 请求安全分类实验

输入一条 HTTP 请求，模型回答两个问题：**是否拦截**（Noul：是／否）和**如果它是攻击，主攻击类别是什么**（Choice：SQL 注入／XSS／路径遍历／其他攻击）。两个问题各有自己的分数分布，每组在页面上都合计 100.00%。Choice 是“已是攻击”条件下的类别判断；当 Noul 选择“否”时，页面不选中攻击类别。模型最高分答案与校准集确定的实验策略阈值分别展示。

这里的“类 Jev”指同一输入回答 Noul 与 Choice 的交互思路，**不声称复现 Jev 未公开的内部模型**。本项目是离线分类和本地试用工具：不会发送或执行输入的 HTTP 请求，也没有接入网络拦截链路。

## 仓库有什么

```text
security_jev_v2/       BERT 双分类头、预处理、训练、校准、评估、本地 HTTP 页面与 API
security_data/         公开来源下载、分组构造、格式化、脱敏和拆分审计
data/v2/               可公开分享的额外脱敏 JSONL 训练／验证／校准／测试数据
tests/                 数据、模型、评估与页面测试
docs/                  本次训练运行记录
artifacts/bert-base-v2/ 模型配置、tokenizer、阈值和实验报告；权重从 Release 附件解压
```

完整 v2 权重文件 `model.pt` 约 411 MB，超过 GitHub 普通文件限制，因此通过 `v0.1.0` GitHub Release 的模型包附件提供，文件名为 `bert-security-v2-1791770771.tar.gz`；其他配置及 tokenizer 同时留在仓库中。

## 环境与安装

本次运行环境是 **macOS、M4 Pro 24 GB、Python 3.12.13、PyTorch 2.14.1、Transformers 5.19.0**，设备使用 MPS。CPU 也能推理，训练会慢很多。请从仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

首次重新训练会从 [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) 下载原版 tokenizer 和预训练 Encoder 权重；**没有从零预训练，也没有继续做 MLM 预训练**。本次完整训练的模型附件已包含推理所需权重和 tokenizer，解压后可离线加载。

## 直接运行已训练模型

从 [v0.1.0 Release](https://github.com/EasonW3300/Jev-WAF-Lite/releases/tag/v0.1.0) 下载模型附件，在仓库根目录解压。附件 SHA-256 为 `44665d6e578ef3fa960955774ebc4bcc8d9d1a1b18339a066578c095b7dc554b`：

```bash
mkdir -p artifacts
tar -xzf bert-security-v2-1791770771.tar.gz -C artifacts
python -m security_jev_v2.web --bundle artifacts/bert-base-v2 --host 127.0.0.1 --port 8766 --device auto
```

浏览器打开 <http://127.0.0.1:8766/>。`auto` 在可用时选择 MPS／CUDA，否则使用 CPU。页面调用 `/api/v2/predict`，返回 Noul 和 Choice 两组分数、条件类别是否适用、实验策略动作、模型版本、截断比例和处理耗时；输入最大 64 KiB。也可用命令行分析保存的请求：

```bash
python -m security_jev_v2.predict --bundle artifacts/bert-base-v2 --request-file request.http --device cpu
```

如果只是查看代码或重新训练，无需下载已训练模型附件。仓库中的 `artifacts/bert-base-v2/test_report.json` 是**原实验的冻结报告**；它不是公开脱敏数据副本上的重算结果。

## 数据来源和格式

| 来源 | 用途与处理 | 许可／版本 |
| --- | --- | --- |
| [open-appsec WAF Comparison Project](https://github.com/openappsec/waf-comparison-project) | 跨站点正常请求模板；每站点最多 100 条。归档中每个浏览文件只读前 5,000 条，超过 50 MiB 的文件跳过，并过滤非主站点请求。 | 合法请求数据 Apache-2.0；归档 SHA-256 在 `data/v2/report.json` |
| [HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset) | SQL 注入、XSS、路径遍历、命令注入 payload；按家族拆分后注入正常请求的可控参数。注入行明确标为**合成**，每站点每类最多 20 条。 | MIT；commit `926670a710283f87c05b554680facf3f9530548c` |
| [OWASP CRS](https://github.com/coreruleset/coreruleset) | 正向规则测试中的攻击候选，特别是命令执行相关样本。 | Apache-2.0；commit `af76ccfcd1ad947e38ef9998bdd464f1a39fef86` |
| 历史 CSIC/CRS 弱标签 | 原实验仅在训练集限额使用；旧测试结果曾被查看，未进入新测试集。 | CSIC 许可未明确核实，发布副本**不包含** 1,100 条 CSIC 行 |

固定种子为 42，先按站点和 payload 家族分组，再以 **60%／10%／10%／20%** 分配训练、验证、校准、测试。同站点、同模板、近重复请求及同 payload 家族不得跨集合。测试集来自 55 个未见主站点；其“正常”仍是公开来源标签，攻击是合成注入，不是目标应用执行后确认的攻击。

每行是一个 JSON 对象，例如：

```json
{"id":"示例ID","state":"GET /search?q=book HTTP/1.1\nHost: <HOST>","block":0,"primary_type":null,"site":"example.test","source":"wcp","origin":"source_legitimate","source_id":"示例来源","template_id":"示例模板","payload_family":null,"pair_id":"示例配对"}
```

`state` 是**唯一模型文本输入**；其余字段用于标签、来源追踪、配对或防泄漏审计。`block` 为 0 或 1；攻击行的 `primary_type` 在 `SQL注入、XSS、路径遍历、其他攻击` 中选一个，正常行是 `null`。未确认主类别的请求不进入监督训练。训练与推理共用 HTTP 规范化：遮盖 Host 和常见凭据字段，最长 512 token，超长请求保留开头和结尾并记录保留比例。

### 训练时数据与公开副本的区别

| 数据集 | 原实验正常 | 原实验四类攻击合计 | 原实验总数 | 仓库发布总数 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 16,662 | 5,854 | **22,516** | **21,416** |
| 验证 | 3,313 | 763 | 4,076 | 4,076 |
| 校准 | 2,558 | 538 | 3,096 | 3,096 |
| 测试 | 5,072 | 1,349 | 6,421 | 6,421 |

原实验训练集的四类分别为 SQL 注入 1,827、XSS 1,737、路径遍历 443、其他攻击 1,847；测试集分别为 391、391、178、389。为避免再公开请求中残留的邮箱、IP、授权令牌和云访问标识，`data/v2/` 又经过 `security_data.public_release` 额外脱敏，并移除许可不明确的 1,100 条历史 CSIC 行。**公开副本与原模型训练数据哈希不同；用它重新训练是一次新实验，不能宣称精确复现下文数字。**新旧哈希、脱敏次数及来源版本见 `data/v2/report.json`；拆分审计见 `data/v2/audit.json`。原始公开归档可通过 `bash security_data/fetch_sources.sh` 下载，下载后可用 `python -m security_data.v2` 和 `python -m security_data.finalize_v2` 重新构造一个不含历史 CSIC 的新数据集。该新构造数据也不会与本仓库冻结的发布副本逐字节相同。

## 如何微调

先校验仓库中的发布数据，再进行 100 步资源试跑和完整训练。请为新实验使用**新输出目录**，不要覆盖随 Release 提供的模型包：

```bash
python -m security_data.audit_v2 --data data/v2
python -m security_jev_v2.train --data data/v2 --output artifacts/pilot-public \
  --model google-bert/bert-base-cased --max-steps 100 --device auto
python -m security_jev_v2.train --data data/v2 --output artifacts/retrained-public \
  --model google-bert/bert-base-cased --max-length 512 \
  --batch-size 4 --accum 4 --epochs 3 \
  --encoder-lr 2e-5 --head-lr 1e-4 --device auto
```

模型共享 BERT 的 `[CLS]` 表示，经 dropout 分别进入 `Linear(768,1)` 的 Noul 头和 `Linear(768,4)` 的 Choice 头。Noul 使用 `BCEWithLogitsLoss`；Choice 只在**攻击且主类别已确认**的样本上使用交叉熵，正常请求不参与类别损失；两项损失等权。编码器和两个新头一起更新，类别头随机初始化。

本次原实验的配置为：物理 batch **4**、梯度累积 **4**（有效 batch 16）、**3 epoch／4,224 次优化更新**、Encoder 学习率 **2e-5**、头部 **1e-4**、AdamW 权重衰减 **0.01**、6% warmup 后线性衰减、梯度裁剪 1.0、梯度检查点与长度分桶、最大 **512 token**。若 M4 Pro 内存不足，可用 `--batch-size 2 --accum 8` 保持有效 batch 16。正式训练从验证集选第 3 轮权重，独立校准集拟合两头分数并确定实验策略阈值 **0.12603**；测试集没有用于选模型或阈值。本机前 100 次更新的中位耗时约 **1.59 秒／步**，采样到的 MPS 内存最大约 **4.49 GB**；该数字不保证其他设备的训练时长或内存。

重新训练的模型包会带上自己的数据 SHA-256、tokenizer、权重、阈值和报告。若要评估**自己新训练的**模型，运行：

```bash
python -m security_jev_v2.evaluate --bundle artifacts/retrained-public \
  --data data/v2 --batch-size 16 --device auto
```

评估器核对模型清单与测试数据哈希，同一模型包已有报告时返回缓存结果。原实验模型的清单记录的是未额外脱敏的数据哈希，不能拿原模型包对 `data/v2/` 伪装成原测试复现。公开测试集和下文结果已经可见，后续调参应准备新的独立测试数据。

## 本次原实验结果

以下数字来自权重和阈值冻结后的**一次原实验测试**：6,421 条请求，其中 5,072 条正常、1,349 条合成攻击。它们**不代表真实生产流量的拦截表现**。

| 指标 | 结果 |
| --- | ---: |
| 是否拦截的总体准确率 | **98.69%**（6,337／6,421） |
| 攻击精确率 | **97.38%** |
| 攻击召回率 | **96.37%**（1,300／1,349） |
| 正常请求误拦率 | **0.69%**（35／5,072；Wilson 95% 区间约 0.50%–0.96%） |
| 拦截头 PR-AUC | **0.9911** |
| 已知攻击 Choice 主类别准确率 | **95.92%** |
| 正确拦截且主类别正确 | **94.89%** |

混淆矩阵 TP／FP／TN／FN 为 **1,300／35／5,037／49**。四类的端到端召回分别为 SQL 注入 **97.44%**、XSS **96.42%**、路径遍历 **97.19%**、其他攻击 **89.72%**。按预设离线门槛（误拦率 ≤1%、总体召回 ≥80%、各达标类别端到端召回 ≥70%）本次通过，但测试攻击均为合成注入。10 条人工试用案例中有一条常见 `UNION SELECT` 注入候选被漏放；测试集约 **26.23%** 的请求超过 512 token 并被首尾截断。细节和校准、分站点指标见 [运行记录](docs/http-security-bert-v2-run-report.md)与 `artifacts/bert-base-v2/test_report.json`。

## 来源、许可与下一步

来源版本和第三方许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。BERT 原版权重及 tokenizer、WCP 正常请求、HttpParamsDataset payload、CRS 测试各归原项目所有。本仓库不收集真实业务流量，不部署自动拦截。要评估真实使用价值，下一步需要人工复核标签、覆盖长请求和更多未见攻击手法，并在新站点上做独立测试。
