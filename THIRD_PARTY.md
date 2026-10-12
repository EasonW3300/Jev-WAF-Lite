# 数据与模型来源

本仓库的发布数据是对以下公开来源的抽样、组合与额外脱敏；原始归档不随仓库再分发。

| 来源 | 用途 | 版本与许可 |
| --- | --- | --- |
| [open-appsec WAF Comparison Project](https://github.com/openappsec/waf-comparison-project) | 跨站点正常 HTTP 请求模板 | 下载归档 SHA-256 见 `data/v2/report.json`；合法请求数据 Apache-2.0 |
| [HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset) | SQL 注入、XSS、路径遍历、命令注入 payload | commit `926670a710283f87c05b554680facf3f9530548c`；MIT |
| [OWASP CRS](https://github.com/coreruleset/coreruleset) | 正向规则测试中的攻击候选 | commit `af76ccfcd1ad947e38ef9998bdd464f1a39fef86`；Apache-2.0 |
| [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) | 原版 tokenizer 与预训练 Encoder 权重 | 模型卡标示 Apache-2.0 |

模型训练时还使用了历史 CSIC 弱标签。其许可未能从原始来源明确核实，**发布数据已剔除这些 1,100 条记录**；模型的训练事实和指标仍如实记录。公开发布副本中任何样本都不应被视为人工验证过的真实攻击或真实业务放行真值。

Apache-2.0 条款文本见 `third_party_licenses/APACHE-2.0.txt`；HttpParamsDataset 的 MIT 条款见 `third_party_licenses/HttpParams-MIT.txt`。来源项目保留各自的版权及商标权。
