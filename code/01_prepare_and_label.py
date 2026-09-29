"""
Step 1: 读取 Financial PhraseBank，构造协变量，调用 DeepSeek 用三套 prompt 标注。
API key 从环境变量 DEEPSEEK_API_KEY 读取，不写入任何文件。
"""
import os, json, time, re, urllib.request, concurrent.futures as cf
import pandas as pd

D = "FinancialPhraseBank-v1.0"
ZIP_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/"
           "resolve/main/data/FinancialPhraseBank-v1.0.zip")
if not os.path.exists(D):     # 数据许可为 CC BY-NC-SA 3.0，这里不重新分发，运行时自动下载
    import zipfile, io
    with urllib.request.urlopen(ZIP_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(".")

def read(fn):
    rows = []
    with open(f"{D}/{fn}", encoding="latin-1") as f:
        for line in f:
            s, lab = line.strip().rsplit("@", 1)
            rows.append((s.strip(), lab.strip()))
    return pd.DataFrame(rows, columns=["text", "human"])

df = read("Sentences_50Agree.txt").drop_duplicates("text").reset_index(drop=True)
s75 = set(read("Sentences_75Agree.txt").text)
sAll = set(read("Sentences_AllAgree.txt").text)
df["agree_all"] = df.text.isin(sAll).astype(int)
df["ambig"] = (~df.text.isin(s75)).astype(int)          # 人工一致性 < 75%：难判句子
df["nwords"] = df.text.str.split().str.len()
df["has_num"] = df.text.str.contains(r"\d").astype(int)
df["id"] = range(len(df))

PROMPTS = {
    # P1：忠实于原始标注手册（投资者视角、只看句内信息）
    "P1": ("You are annotating financial news sentences. Taking the viewpoint of an investor, "
           "decide whether the sentence is likely to have a positive, negative, or neutral influence "
           "on the company's stock price. Use only information explicitly stated in the sentence. "
           "Answer with one word: positive, negative, or neutral."),
    # P2：一般性的情感分析措辞
    "P2": ("What is the sentiment of the following sentence? "
           "Answer with one word: positive, negative, or neutral."),
    # P3：“是不是好消息”，要求模型更果断
    "P3": ("Read the following business news sentence. Is it good news, bad news, or neither for the "
           "company mentioned? Avoid 'neutral' unless the sentence contains no evaluative information. "
           "Answer with one word: positive, negative, or neutral."),
}

KEY = os.environ.get("DEEPSEEK_API_KEY", "")   # 已有 llm_P*.csv 时不会调用 API
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

def call(prompt, text, retries=5):
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": 5,
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": text}],
    }).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            ans = out["choices"][0]["message"]["content"].strip().lower()
            m = re.search(r"positive|negative|neutral", ans)
            return (m.group(0) if m else "PARSE_ERR:" + ans), out.get("model", "")
        except Exception as e:
            time.sleep(2 ** k)
    return "API_ERR", ""

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(df)
    sub = df.head(n).copy()
    for p, prompt in PROMPTS.items():
        cache = f"llm_{p}.csv"
        done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["id", p, "model"])
        todo = sub[~sub.id.isin(done.id)]
        print(p, "todo", len(todo), flush=True)
        res = []
        with cf.ThreadPoolExecutor(32) as ex:
            futs = {ex.submit(call, prompt, t): i for i, t in zip(todo.id, todo.text)}
            for j, f in enumerate(cf.as_completed(futs)):
                lab, mdl = f.result()
                res.append((futs[f], lab, mdl))
                if (j + 1) % 500 == 0:
                    print(p, j + 1, flush=True)
        done = pd.concat([done, pd.DataFrame(res, columns=["id", p, "model"])])
        done.to_csv(cache, index=False)
    df.to_csv("corpus.csv", index=False)
    print("date:", time.strftime("%Y-%m-%d"))
