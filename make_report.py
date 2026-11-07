"""把一份有星等的評論 CSV 整理成一頁靜態洞察報告（docs/index.html）。

用法：python make_report.py 評論.csv "App 名稱"
CSV 欄位：date, source, rating, text
"""
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analysis import assign_topics, classify_sentiment, load_sentiment_lexicon, load_topic_rules

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "index.html"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("text") and r.get("rating")]
    for r in rows:
        r["rating"] = int(r["rating"])
    return rows


def quarter(day: str) -> str:
    return f"{day[:4]} Q{(int(day[5:7]) - 1) // 3 + 1}"


def lexicon_accuracy(rows: list[dict]) -> tuple[float, int]:
    """拿星等當答案，檢查詞典判斷的準確率（只看 1–2 星與 4–5 星）。

    調整詞典時只看了奇數列的評論，這裡只用偶數列驗證，避免高估。"""
    lexicon = load_sentiment_lexicon(ROOT / "data" / "sentiment_lexicon.json")
    checked = correct = 0
    for r in rows[0::2]:
        truth = "negative" if r["rating"] <= 2 else "positive" if r["rating"] >= 4 else None
        if truth:
            checked += 1
            correct += classify_sentiment(r["text"], lexicon).label == truth
    return correct / checked, checked


def short(text: str, n: int = 34) -> str:
    text = text.replace("。", "：", 1)
    return text if len(text) <= n else text[:n] + "…"


def bar_svg(items: list[tuple[str, float, str]], unit: str, max_value: float, width: int = 640) -> str:
    """水平長條圖：items = [(標籤, 數值, 顏色)]。"""
    label_w, row_h = 230, 34
    bar_max = width - label_w - 70
    parts = [f'<svg viewBox="0 0 {width} {len(items) * row_h + 6}" role="img">']
    for i, (label, value, color) in enumerate(items):
        y = i * row_h + 6
        w = max(2, value / max_value * bar_max)
        parts.append(f'<text x="0" y="{y + 17}" class="lbl">{html.escape(label)}</text>'
                     f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="18" rx="5" fill="{color}"/>'
                     f'<text x="{label_w + w + 8:.1f}" y="{y + 18}" class="val">{value:g}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_svg(points: list[tuple[str, float]], width: int = 640, height: int = 220) -> str:
    """平均星等折線圖（1–5 星）。"""
    left, right, top, bottom = 36, 40, 16, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / max(1, len(points) - 1)
    def y_of(v): return top + (5 - v) / 4 * plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for star in range(1, 6):
        y = y_of(star)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="0" y="{y + 4:.1f}" class="axis">{star}★</text>')
    coords = [(left + i * step, y_of(v)) for i, (_, v) in enumerate(points)]
    parts.append('<polyline fill="none" stroke="#e8743b" stroke-width="3" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + '"/>')
    for (label, value), (x, y) in zip(points, coords):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#e8743b"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="axis" text-anchor="middle">{label.replace(" ", "")}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build(csv_path: Path, app_name: str) -> str:
    rows = load_rows(csv_path)
    rules = load_topic_rules(ROOT / "data" / "topic_rules.json")
    for r, topic in zip(rows, assign_topics([r["text"] for r in rows], rules)):
        r["topic"] = topic

    n = len(rows)
    dates = sorted(r["date"] for r in rows)
    avg = sum(r["rating"] for r in rows) / n
    neg_share = sum(r["rating"] <= 2 for r in rows) / n
    accuracy, checked = lexicon_accuracy(rows)

    # 各主題：則數、平均星等、負評比例
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    topic_stats = sorted(
        ((name, len(items), sum(i["rating"] for i in items) / len(items),
          sum(i["rating"] <= 2 for i in items) / len(items)) for name, items in by_topic.items()),
        key=lambda t: -t[1])

    # 每季平均星等
    by_q = defaultdict(list)
    for r in rows:
        by_q[quarter(r["date"])].append(r["rating"])
    trend = [(q, round(sum(v) / len(v), 2)) for q, v in sorted(by_q.items()) if len(v) >= 20]

    # 每年平均星等
    year_avg = {}
    for y in sorted({r["date"][:4] for r in rows}):
        v = [r["rating"] for r in rows if r["date"].startswith(y)]
        year_avg[y] = sum(v) / len(v)

    # 系統不穩占比：每年
    years = sorted({r["date"][:4] for r in rows})
    year_share = []
    for y in years:
        items = [r for r in rows if r["date"].startswith(y)]
        hit = sum(r["topic"].startswith("系統不穩") for r in items)
        year_share.append((y, round(hit / len(items) * 100), len(items)))

    def quotes(name: str, k: int = 2) -> list[str]:
        """挑負評當例句；優先挑含有該主題前三個關鍵字的，確保例句真的在講這件事。"""
        keys = rules.get(name, [])[:3]
        items = [r for r in by_topic.get(name, []) if r["rating"] <= 2 and 8 <= len(r["text"]) <= 80]
        items.sort(key=lambda r: not any(key in r["text"] for key in keys))
        return [short(r["text"]) for r in items[:k]]

    referral = next((t for t in topic_stats if t[0].startswith("推薦碼")), None)
    stable = next(t for t in topic_stats if t[0].startswith("系統不穩"))
    login = next(t for t in topic_stats if t[0].startswith("登入"))
    reward = next(t for t in topic_stats if t[0].startswith("回饋"))

    topic_bars = bar_svg([(name, cnt, "#e8743b" if neg >= .7 else "#4f7cac") for name, cnt, _, neg in topic_stats],
                         " 則", topic_stats[0][1])
    neg_bars = bar_svg([(name, round(neg * 100), "#e8743b" if neg >= .7 else "#9bb3cc")
                        for name, cnt, _, neg in sorted(topic_stats, key=lambda t: -t[3]) if cnt >= 10],
                       "%", 100)
    year_bars = bar_svg([(f"{y} 年（{total} 則）", share, "#e8743b") for y, share, total in year_share], "%", 100)

    findings = [
        ("系統穩定性是最大痛點，2026 年比重明顯升高",
         f"「{stable[0]}」共 {stable[1]} 則，是所有主題中最多的，平均只有 {stable[2]:.1f} 星。"
         f"它占當年評論的比例在 {year_share[0][0]}、{year_share[1][0]} 年約兩成（{year_share[0][1]}%、{year_share[1][1]}%），"
         f"{year_share[-1][0]} 年升到 {year_share[-1][1]}%。",
         quotes(stable[0]),
         "版本上線前加強相容性測試、採分批更新；在 App 內與商店頁即時公告維修時間與已知問題。"),
        ("登入與身分驗證最傷口碑",
         f"「{login[0]}」有 {login[1]} 則，平均 {login[2]:.1f} 星、負評比例 {login[3]:.0%}，是星等最低的主要主題。"
         "常見的是重新驗證流程看不懂、等待審核太久。",
         quotes(login[0]),
         "在驗證頁用圖示說明拍攝方式與預計審核時間，並主動推播審核進度，減少「不知道卡在哪」的焦慮。"),
        ("回饋與優惠的變動，需要更透明的溝通",
         f"「{reward[0]}」有 {reward[1]} 則、平均 {reward[2]:.1f} 星，多數在抱怨回饋變少或規則不清楚。",
         quotes(reward[0]),
         "調整回饋前先公告比較表與生效日，並在 App 內提供「我的回饋怎麼算」說明頁。"),
    ]
    if referral:
        findings.append((
            "高分評論有一部分其實是推薦碼貼文",
            f"「{referral[0]}」有 {referral[1]} 則，平均高達 {referral[2]:.1f} 星，內容多是貼邀請碼換獎勵。"
            "評估真實滿意度時，應把這類評論分開看，否則會高估口碑。",
            [], "內部追蹤口碑時排除推薦碼貼文；行銷上可另外分析推薦活動帶來的實際註冊與留存。"))

    finding_html = ""
    for i, (title, body, qs, action) in enumerate(findings, 1):
        q_html = "".join(f"<blockquote>「{html.escape(q)}」</blockquote>" for q in qs)
        finding_html += (f'<article class="finding"><div class="num">{i:02d}</div><div><h3>{html.escape(title)}</h3>'
                         f'<p>{html.escape(body)}</p>{q_html}<p class="action"><b>建議做法</b>{html.escape(action)}</p></div></article>')

    rule_count = len(rules)
    return TEMPLATE.format(
        app=html.escape(app_name), n=n, start=dates[0], end=dates[-1], avg=f"{avg:.2f}", neg=f"{neg_share:.0%}",
        topics=rule_count, other=f"{len(by_topic.get('其他', [])) / n:.0%}",
        acc=f"{accuracy:.0%}", checked=checked,
        trend=line_svg(trend), year_text="、".join(f"{y} 年 {v:.2f} 星" for y, v in year_avg.items()),
        topic_bars=topic_bars, neg_bars=neg_bars, year_bars=year_bars, findings=finding_html)


TEMPLATE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{app} App 評論洞察報告</title>
<style>
:root{{--ink:#1d2733;--muted:#66717d;--line:#e3e7ec;--paper:#f7f8fa;--card:#fff;--accent:#e8743b;--blue:#4f7cac}}
@media (prefers-color-scheme:dark){{:root{{--ink:#e7ebf0;--muted:#9aa5b1;--line:#2b3440;--paper:#12171d;--card:#1a2129}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 "Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:48px 16px 64px}}
.eyebrow{{font-size:12px;letter-spacing:.14em;color:var(--accent);font-weight:700}}
h1{{font-size:34px;line-height:1.3;margin:6px 0 10px}}h2{{font-size:22px;margin:44px 0 12px}}h3{{margin:0 0 6px;font-size:19px}}
.lead{{color:var(--muted)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.kpi b{{display:block;font-size:28px}}.kpi span{{color:var(--muted);font-size:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:12px 0}}
svg{{width:100%;height:auto;display:block}}svg .lbl,svg .val,svg .axis{{fill:var(--ink);font-size:13px}}svg .axis{{fill:var(--muted)}}svg .grid{{stroke:var(--line)}}
.finding{{display:flex;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:14px 0}}
.num{{font-size:26px;font-weight:700;color:var(--accent);min-width:40px}}
blockquote{{margin:8px 0;padding:6px 12px;border-left:3px solid var(--accent);color:var(--muted);font-size:15px}}
.action{{background:rgba(232,116,59,.08);border-radius:8px;padding:8px 12px}}.action b{{margin-right:8px;color:var(--accent)}}
.note{{font-size:14px;color:var(--muted)}}ul{{padding-left:20px}}
footer{{margin-top:40px;font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px}}
</style></head><body><main>
<div class="eyebrow">CUSTOMER VOICE · INSIGHT REPORT</div>
<h1>{app} App Store 評論洞察報告</h1>
<p class="lead">以「顧客回饋洞察平台」分析 App Store 台灣區最近 {n} 則公開評論（{start} 至 {end}），找出最值得優先處理的使用者痛點。</p>
<div class="kpis">
<div class="kpi"><b>{n}</b><span>則公開評論</span></div>
<div class="kpi"><b>{avg} ★</b><span>平均星等</span></div>
<div class="kpi"><b>{neg}</b><span>為 1–2 星負評</span></div>
<div class="kpi"><b>{topics}</b><span>個主題，「其他」僅占 {other}</span></div>
</div>

<h2>星等趨勢：每季平均</h2>
<div class="card">{trend}<p class="note">年平均星等：{year_text}。每季至少 20 則評論才畫進圖中。</p></div>

<h2>大家在抱怨什麼：主題分布</h2>
<div class="card">{topic_bars}<p class="note">橘色＝負評比例 70% 以上的主題。</p></div>

<h2>哪些主題最負面：負評比例</h2>
<div class="card">{neg_bars}<p class="note">只列出 10 則以上的主題。</p></div>

<h2>系統不穩的比重：2026 年明顯升高</h2>
<div class="card">{year_bars}<p class="note">「系統不穩（閃退、維修、打不開）」占當年評論的比例。</p></div>

<h2>四個關鍵發現與建議</h2>
{findings}

<h2>方法與限制</h2>
<div class="card"><ul>
<li><b>資料：</b>透過 Apple 公開的評論 RSS 取得台灣區最近 {n} 則評論。App Store 只提供最近的評論，且留評者通常是體驗特別好或特別差的人，不代表全體使用者。</li>
<li><b>主題：</b>用可編輯的關鍵字規則分類，每則評論歸到命中最多關鍵字的主題；平台也提供 TF-IDF＋KMeans 自動分群，用來探索規則以外的新主題。</li>
<li><b>情緒：</b>有星等時直接用星等。沒有星等時改用中文情緒詞典並處理否定詞；以星等當答案驗證，在沒有用來調整詞典的 {checked} 則評論上，詞典判斷的準確率約 {acc}（原始詞典約 37%）。</li>
<li><b>用途：</b>這是作品集展示用的分析，評論內容只摘錄片段，結論僅供參考，不代表該公司的官方數據。</li>
</ul></div>
<footer>製作：林伯諺 · 工具：Python、Flask、SQLite、scikit-learn、jieba · 原始碼與平台說明見 GitHub</footer>
</main></body></html>"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("用法：python make_report.py 評論.csv \"App 名稱\"")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(Path(sys.argv[1]), sys.argv[2]), encoding="utf-8")
    print("已輸出", OUT)
