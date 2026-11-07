"""情緒判斷與主題分群。

情緒：用詞典比對正負向詞，並處理「不好用」「不太方便」這類否定。
主題：先用 jieba 斷詞、去除停用詞，再用 TF-IDF + KMeans 把相似的回饋分在同一群。
"""
import json
import re
from pathlib import Path

import jieba
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

from models import SentimentLexicon, SentimentResult, TopicResult

jieba.setLogLevel(60)  # 關掉 jieba 載入字典時的訊息

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------- 情緒 ----------

def load_sentiment_lexicon(path: Path) -> SentimentLexicon:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return SentimentLexicon(
        positive=set(data.get("positive", [])),
        negative=set(data.get("negative", [])),
        negation=set(data.get("negation", [])),
    )


def _find_spans(text: str, words: set[str]) -> list[tuple[int, int, str]]:
    """找出每個詞在文字中出現的位置 (開始, 結束, 詞)。"""
    spans = []
    for word in words:
        for match in re.finditer(re.escape(word), text):
            spans.append((match.start(), match.end(), word))
    return spans


def classify_sentiment(text: str, lexicon: SentimentLexicon) -> SentimentResult:
    """比對正負向詞，數量多的一方決定情緒；一樣多就是中立。

    兩個規則：
    1. 正向詞如果包含在負向詞裡（例如「不清楚」裡的「清楚」），只算負向。
    2. 正向詞前面兩個字內出現否定詞（例如「不太方便」），改算成負向。
    """
    negative_spans = _find_spans(text, lexicon.negative)
    positive_hits, negative_hits = [], [word for _, _, word in negative_spans]

    for start, end, word in _find_spans(text, lexicon.positive):
        inside_negative = any(s <= start and end <= e for s, e, _ in negative_spans)
        if inside_negative:
            continue
        before = text[max(0, start - 2):start]
        if any(neg in before for neg in lexicon.negation):
            negative_hits.append(before + word)
        else:
            positive_hits.append(word)

    if len(positive_hits) > len(negative_hits):
        label = "positive"
    elif len(negative_hits) > len(positive_hits):
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(label, sorted(positive_hits + negative_hits))


# ---------- 主題 ----------

def _load_stopwords() -> set[str]:
    path = DATA_DIR / "stopwords.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    """斷詞後只留下有意義的詞：去掉停用詞、單一字、純數字與標點。"""
    words = []
    for word in jieba.lcut(text.lower()):
        word = word.strip()
        if len(word) < 2 or word in stopwords or word.isdigit():
            continue
        if not re.search(r"[\u4e00-\u9fffa-z]", word):
            continue
        words.append(word)
    return words


def cluster_feedback(texts: list[str], cluster_count: int = 5) -> TopicResult:
    """把回饋分成 cluster_count 群，並回傳每群最具代表性的 5 個詞。"""
    if not texts:
        return TopicResult([], {})

    stopwords = _load_stopwords()
    vectorizer = TfidfVectorizer(
        tokenizer=lambda t: _tokenize(t, stopwords),
        token_pattern=None,
        lowercase=False,
        min_df=2 if len(texts) >= 20 else 1,
        max_df=0.5,
        max_features=3000,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # 文字太少、斷不出任何詞
        return TopicResult([0] * len(texts), {0: []})

    distinct_rows = {(row.indices.tobytes(), row.data.tobytes())
                     for row in (matrix.getrow(i) for i in range(matrix.shape[0])) if row.nnz}
    k = max(1, min(cluster_count, len(distinct_rows), len(texts)))
    if k == 1:
        labels = [0] * len(texts)
        centroids = [matrix.mean(axis=0).A1]
    else:
        # 短評論的詞向量很稀疏，先用 SVD 壓成「語意」向量再分群，群的大小會比較平均
        dims = max(2, min(100, matrix.shape[1] - 1, matrix.shape[0] - 1))
        reduced = Normalizer(copy=False).fit_transform(
            TruncatedSVD(n_components=dims, random_state=42).fit_transform(matrix))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(reduced).tolist()
        # 每群的代表詞：用原本的 TF-IDF 算群內平均，取分數最高的詞
        centroids = [matrix[[i for i, label in enumerate(labels) if label == c]].mean(axis=0).A1
                     for c in range(k)]

    terms = vectorizer.get_feature_names_out()
    top_terms = {i: [terms[j] for j in centroids[i].argsort()[::-1][:5]] for i in range(k)}
    return TopicResult(labels, top_terms)


# ---------- 規則分類（有名字的主題） ----------

OTHER_TOPIC = "其他"


def load_topic_rules(path: Path) -> dict[str, list[str]]:
    """讀取主題規則；以底線開頭的欄位是說明文字，略過。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return {name: words for name, words in data.items() if not name.startswith("_")}


def assign_topics(texts: list[str], rules: dict[str, list[str]]) -> list[str]:
    """每則回饋分到命中最多關鍵字的主題；同分時取規則檔裡排前面的。"""
    topics = []
    for text in texts:
        lowered = text.lower()
        best_name, best_hits = OTHER_TOPIC, 0
        for name, keywords in rules.items():
            hits = sum(1 for word in keywords if word.lower() in lowered)
            if hits > best_hits:
                best_name, best_hits = name, hits
        topics.append(best_name)
    return topics
