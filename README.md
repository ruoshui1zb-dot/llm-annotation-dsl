# LLM 标注的下游纠偏：DSL 与 PPI 实验代码

配套推文：《换 prompt 系数就变号？用 DSL 纠偏》（李忠彬，连享会）

## 文件

| 文件 | 内容 |
|---|---|
| `code/01_prepare_and_label.py` | 下载 Financial PhraseBank，构造协变量，调用 DeepSeek 用三套 prompt 标注 |
| `code/02_analysis.py` | 标注质量、直接估计、DSL / PPI++ 校正、1,000 次重复抽样 |
| `code/03_figures.py` | 生成推文中的 3 张图 |
| `code/llm_P1.csv` `llm_P2.csv` `llm_P3.csv` | DeepSeek 标注结果（`id` 对应去重后句子的顺序） |
| `code/quality.csv` `mc_n300.csv` `mc_curve.csv` | 分析输出 |
| `figs/` | 推文中的 3 张图 |

## 运行

```bash
cd code
pip install -r requirements.txt

# 1. 标注（已提供标注结果，可跳过；重新标注需要自己的 DeepSeek API key）
export DEEPSEEK_API_KEY=你的key
python 01_prepare_and_label.py

# 若跳过第 1 步，仍需运行一次以下载数据、生成 corpus.csv：
#   已有 llm_P*.csv 时脚本只会补标缺失的句子，不会重复调用 API

# 2. 分析与作图
python 02_analysis.py
python 03_figures.py
```

## 实验设定

- 数据：Financial PhraseBank v1.0，`Sentences_50Agree`，去重后 4,838 条；许可 CC BY-NC-SA 3.0（Malo et al., 2014），本包不含原始数据，运行时自动下载
- 模型：DeepSeek API `deepseek-chat`（API 返回 `deepseek-flash`），`temperature = 0`，`max_tokens = 5`
- 调用日期：2026-09-29
- 1 条输出无法解析（P3），按中性处理
- 情形 B 的结果变量 `ret` 为已知参数的模拟变量，随机种子 2026
