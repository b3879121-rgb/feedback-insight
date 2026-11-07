from models import TopicStats, ScoredTopic

def score_topics(topic_stats: list[TopicStats]) -> list[ScoredTopic]:
    if not topic_stats: return []
    max_count = max(t.count for t in topic_stats) or 1
    dated = any(t.recent_growth is not None for t in topic_stats)
    growths = [max(0, t.recent_growth or 0) for t in topic_stats]
    max_growth = max(growths, default=0)
    out = []
    for t in topic_stats:
        volume = t.count / max_count * 100
        negative = t.negative_share * 100
        growth = (max(0, t.recent_growth or 0) / max_growth * 100) if dated and max_growth else 0
        score = .4*volume + .4*negative + .2*growth if dated else .5*volume + .5*negative
        out.append(ScoredTopic(t.topic_id, round(score,1), {"volume":round(volume,1),"negative_share":round(negative,1),"growth":round(growth,1)}))
    return sorted(out, key=lambda x:x.score, reverse=True)
