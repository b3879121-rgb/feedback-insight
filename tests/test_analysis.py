"""基本測試：python -m pytest 或 python tests/test_analysis.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import DATA_DIR, assign_topics, classify_sentiment, load_sentiment_lexicon, load_topic_rules

LEXICON = load_sentiment_lexicon(DATA_DIR / "sentiment_lexicon.json")
RULES = load_topic_rules(DATA_DIR / "topic_rules.json")


def test_negation_turns_positive_into_negative():
    assert classify_sentiment("更新後不太方便", LEXICON).label == "negative"


def test_positive_inside_negative_word_is_not_counted():
    assert classify_sentiment("說明不清楚", LEXICON).label == "negative"


def test_plain_positive():
    assert classify_sentiment("介面簡單又好用", LEXICON).label == "positive"


def test_rule_topics():
    topics = assign_topics(["更新後一直閃退", "實名認證等很久", "今天天氣很好"], RULES)
    assert topics[0].startswith("系統不穩")
    assert topics[1] == "登入與身分驗證"
    assert topics[2] == "其他"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("通過", name)
