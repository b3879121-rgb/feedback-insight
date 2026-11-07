from dataclasses import dataclass, field
from datetime import date
from typing import Literal

@dataclass
class FeedbackInput:
    date: date | None
    source: str
    text: str
    rating: int | None = None  # 選填：1–5 星評分

@dataclass
class ImportIssue:
    row_number: int
    field: str
    message: str

@dataclass
class ImportResult:
    sha256: str
    rows: list[FeedbackInput] = field(default_factory=list)
    issues: list[ImportIssue] = field(default_factory=list)

@dataclass
class TopicResult:
    labels: list[int]
    top_terms: dict[int, list[str]]

@dataclass
class SentimentLexicon:
    positive: set[str]
    negative: set[str]
    negation: set[str] = field(default_factory=set)

@dataclass
class SentimentResult:
    label: Literal["positive", "neutral", "negative"]
    matched_terms: list[str]

@dataclass
class TopicStats:
    topic_id: int
    count: int
    negative_share: float
    recent_growth: float | None
    top_terms: list[str]
    examples: list[str]

@dataclass
class ScoredTopic:
    topic_id: int
    score: float
    components: dict[str, float]

@dataclass
class AnalysisSummary:
    batch_id: int
    feedback_count: int
    topic_count: int
    sentiment_counts: dict[str, int]
    priority_topics: list[ScoredTopic]

@dataclass
class FeedbackView:
    id: int
    date: date | None
    source: str
    text: str
    topic_auto: str
    topic_manual: str | None
    sentiment_auto: Literal["positive", "neutral", "negative"]
    sentiment_manual: Literal["positive", "neutral", "negative"] | None

@dataclass
class DashboardSummary:
    feedback_count: int
    topic_count: int
    sentiment_counts: dict[str, int]
    priority_topics: list[ScoredTopic]
