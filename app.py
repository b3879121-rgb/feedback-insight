"""顧客回饋洞察平台：Flask 網頁。

流程：匯入 CSV → 預覽 → 寫入資料庫 → 分析（主題＋情緒）→ 儀表板／匯出 CSV／列印摘要。
"""
import csv
import io
import json
import os
import secrets
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import (Flask, Response, abort, flash, g, redirect, render_template,
                   request, send_file, url_for)

from analysis import (assign_topics, classify_sentiment, cluster_feedback,
                      load_sentiment_lexicon, load_topic_rules)
from db import connect, init_db
from ingest import parse_feedback_csv
from models import TopicStats
from scoring import score_topics

SENTIMENTS = ("positive", "neutral", "negative")
TOKEN_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


# ---------- 共用的小工具 ----------

def topic_of(row) -> str:
    """人工標籤優先，沒有才用自動結果。"""
    return row["topic_manual"] or row["topic_auto"]


def sentiment_of(row) -> str:
    return row["sentiment_manual"] or row["sentiment_auto"]


def parse_day(value: str | None) -> date | None:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def sentiment_from_rating(rating: int) -> str:
    """有星等時直接用星等：1–2 星負向、3 星中性、4–5 星正向。"""
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return "neutral"


def recent_growth(days: list[date], latest: date | None) -> float | None:
    """最近 14 天的則數，相對前 14 天成長多少（只算增加）。沒有日期就回傳 None。"""
    if latest is None:
        return None
    recent = sum(1 for d in days if latest - timedelta(days=13) <= d <= latest)
    previous = sum(1 for d in days if latest - timedelta(days=27) <= d < latest - timedelta(days=13))
    return max(0, (recent - previous) / max(1, previous))


def group_by_topic(rows) -> dict[str, list]:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(topic_of(row), []).append(row)
    return groups


def score_groups(groups: dict[str, list]) -> dict[str, dict]:
    """幫每個主題算出：則數、負向比例、近期成長、優先分數。"""
    all_days = [parse_day(r["feedback_date"]) for items in groups.values() for r in items if r["feedback_date"]]
    latest = max(all_days) if all_days else None
    names = list(groups)
    stats = []
    for index, name in enumerate(names):
        items = groups[name]
        negative = sum(1 for r in items if sentiment_of(r) == "negative")
        days = [parse_day(r["feedback_date"]) for r in items if r["feedback_date"]]
        stats.append(TopicStats(index, len(items), negative / len(items), recent_growth(days, latest), [], []))
    result = {}
    for scored in score_topics(stats):
        name = names[scored.topic_id]
        result[name] = {"score": scored.score, "components": scored.components,
                        "count": stats[scored.topic_id].count,
                        "negative_share": stats[scored.topic_id].negative_share}
    return result


def bar_chart(counts: dict[str, int]) -> list[dict]:
    top = max(counts.values(), default=1) or 1
    return [{"label": label, "count": n, "width": round(n / top * 100)} for label, n in counts.items()]


# ---------- Flask 應用 ----------

def create_app(database_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("APP_SECRET", "local-feedback-insight")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    project_dir = Path(__file__).resolve().parent
    app.config["PROJECT_DIR"] = project_dir
    app.config["DATABASE"] = database_path or str(project_dir / "instance" / "feedback.sqlite3")
    app.config["STAGING"] = Path(app.config["DATABASE"]).parent / "staging"
    app.config["STAGING"].mkdir(parents=True, exist_ok=True)
    with connect(app.config["DATABASE"]) as conn:
        init_db(conn)

    def db() -> sqlite3.Connection:
        if "feedback_db" not in g:
            g.feedback_db = connect(app.config["DATABASE"])
        return g.feedback_db

    @app.teardown_appcontext
    def close_db(error=None):
        conn = g.pop("feedback_db", None)
        if conn is not None:
            conn.close()

    def cleanup_staging():
        """刪掉超過一天、沒有確認匯入的預覽檔。"""
        cutoff = datetime.now().timestamp() - 86400
        for path in app.config["STAGING"].glob("*.json"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def load_batch(batch_id: int):
        conn = db()
        batch = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not batch:
            abort(404)
        rows = conn.execute(
            "SELECT * FROM feedback WHERE batch_id=? ORDER BY feedback_date DESC, id DESC", (batch_id,)).fetchall()
        topics = conn.execute(
            "SELECT * FROM topic_summaries WHERE analysis_run_id="
            "(SELECT id FROM analysis_runs WHERE batch_id=?) ORDER BY priority_score DESC", (batch_id,)).fetchall()
        return batch, rows, topics

    def terms_for(topics, name: str) -> list[str]:
        saved = next((t for t in topics if t["label_auto"] == name), None)
        return json.loads(saved["top_terms_json"]) if saved else []

    # ----- 首頁與匯入 -----

    @app.get("/")
    def home():
        batches = db().execute("SELECT * FROM batches ORDER BY id DESC").fetchall()
        return render_template("dashboard.html", batches=batches, active=None)

    @app.get("/demo.csv")
    def demo_csv():
        return send_file(project_dir / "data" / "demo_feedback.csv", as_attachment=True,
                         download_name="demo_feedback.csv", mimetype="text/csv")

    @app.post("/import/preview")
    def preview():
        file = request.files.get("file")
        if not file or not file.filename:
            flash("請先選擇 CSV 檔案", "error")
            return redirect(url_for("home"))
        cleanup_staging()
        try:
            result = parse_feedback_csv(file.stream)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("home"))
        if not result.rows:
            flash("沒有可匯入的有效回饋", "error")
            return redirect(url_for("home"))
        if db().execute("SELECT 1 FROM batches WHERE sha256=?", (result.sha256,)).fetchone():
            flash("這份 CSV 已匯入過，系統已阻止重複批次。", "error")
            return redirect(url_for("home"))

        token = secrets.token_urlsafe(24)
        payload = {
            "filename": file.filename,
            "sha256": result.sha256,
            "rows": [{"date": r.date.isoformat() if r.date else None, "source": r.source,
                      "text": r.text, "rating": r.rating} for r in result.rows],
            "issues": [issue.__dict__ for issue in result.issues],
        }
        (app.config["STAGING"] / f"{token}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return render_template("import_preview.html", token=token, payload=payload)

    @app.post("/import/commit")
    def commit():
        cleanup_staging()
        token = request.form.get("token", "")
        staged = app.config["STAGING"] / f"{token}.json"
        if not token or not set(token) <= TOKEN_CHARS or not staged.exists():
            flash("預覽已過期或無效，請重新上傳", "error")
            return redirect(url_for("home"))
        payload = json.loads(staged.read_text(encoding="utf-8"))
        try:
            with db() as conn:
                cur = conn.execute("INSERT INTO batches(filename, sha256, record_count) VALUES(?,?,?)",
                                   (Path(payload["filename"]).name, payload["sha256"], len(payload["rows"])))
                batch_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO feedback(batch_id, feedback_date, source, body, rating) VALUES(?,?,?,?,?)",
                    [(batch_id, r["date"], r["source"], r["text"], r.get("rating")) for r in payload["rows"]])
        except sqlite3.IntegrityError:
            app.logger.exception("Import failed: integrity error")
            flash("匯入失敗：這份檔案可能已經匯入。原資料未變更。", "error")
            return redirect(url_for("home"))
        except sqlite3.Error:
            app.logger.exception("Import failed")
            flash("資料庫匯入失敗，原資料未變更。", "error")
            return redirect(url_for("home"))
        staged.unlink(missing_ok=True)
        flash(f"已匯入 {len(payload['rows'])} 筆回饋", "success")
        return redirect(url_for("batch", batch_id=batch_id))

    # ----- 儀表板 -----

    @app.get("/batches/<int:batch_id>")
    def batch(batch_id):
        b, rows, topics = load_batch(batch_id)
        batches = db().execute("SELECT * FROM batches ORDER BY id DESC").fetchall()

        groups = group_by_topic(rows)
        effective_topics = {name: len(items) for name, items in groups.items()}
        sentiments = {s: 0 for s in SENTIMENTS}
        for row in rows:
            sentiments[sentiment_of(row)] += 1
        sentiment_evidence = {row["id"]: json.loads(row["sentiment_terms_auto"] or "[]") for row in rows}

        scores = score_groups(groups)
        priority_topics = [
            {"label": name, "score": s["score"], "components": s["components"],
             "terms": terms_for(topics, name), "examples": groups[name][:5]}
            for name, s in sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)]

        # 日期跨度超過 60 天時改用「月」彙整，避免一天一列太長
        days = sorted(r["feedback_date"] for r in rows if r["feedback_date"])
        by_month = bool(days) and (parse_day(days[-1]) - parse_day(days[0])).days > 60
        date_counts: dict[str, int] = {}
        for day in days:
            key = day[:7] if by_month else day
            date_counts[key] = date_counts.get(key, 0) + 1
        topic_chart = bar_chart(dict(sorted(effective_topics.items(), key=lambda kv: -kv[1])))
        date_chart = bar_chart(dict(sorted(date_counts.items())))
        analysis_run = db().execute("SELECT * FROM analysis_runs WHERE batch_id=?", (batch_id,)).fetchone()

        return render_template(
            "batch.html", b=b, rows=rows, topics=topics, batches=batches,
            effective_topics=effective_topics, sentiments=sentiments,
            dated=sum(1 for r in rows if r["feedback_date"]), filters=request.args,
            priority_topics=priority_topics, date_counts=date_counts,
            sentiment_evidence=sentiment_evidence, topic_chart=topic_chart, date_chart=date_chart,
            topic_chart_height=max(80, len(topic_chart) * 36), date_chart_height=max(80, len(date_chart) * 30),
            has_rules=bool(load_topic_rules(project_dir / "data" / "topic_rules.json")),
            rated=sum(1 for r in rows if r["rating"]), analysis_run=analysis_run)

    @app.post("/batches/<int:batch_id>/analyze")
    def analyze(batch_id):
        method = request.form.get("method", "rules")
        try:
            clusters = int(request.form.get("clusters", 5))
        except (TypeError, ValueError):
            clusters = 0
        if method == "kmeans" and not 2 <= clusters <= 8:
            flash("主題群數需介於 2 到 8 群。", "error")
            return redirect(url_for("batch", batch_id=batch_id))

        conn = db()
        rows = conn.execute("SELECT * FROM feedback WHERE batch_id=?", (batch_id,)).fetchall()
        if not rows:
            abort(404)
        texts = [r["body"] for r in rows]

        # 1. 主題：規則分類（有名字）或自動分群（主題 1、主題 2…）
        rules = load_topic_rules(project_dir / "data" / "topic_rules.json")
        top_terms: dict[str, list[str]] = {}
        if method == "rules" and rules:
            topic_names = assign_topics(texts, rules)
            for name in set(topic_names):
                top_terms[name] = rules.get(name, [])[:5]
        else:
            method = "kmeans"
            result = cluster_feedback(texts, clusters)
            topic_names = [f"主題 {label + 1}" for label in result.labels]
            top_terms = {f"主題 {i + 1}": words for i, words in result.top_terms.items()}

        # 2. 情緒：有星等用星等，沒有就用詞典判斷
        lexicon = load_sentiment_lexicon(project_dir / "data" / "sentiment_lexicon.json")
        with conn:
            for row, topic in zip(rows, topic_names):
                found = classify_sentiment(row["body"], lexicon)
                label = sentiment_from_rating(row["rating"]) if row["rating"] else found.label
                conn.execute("UPDATE feedback SET topic_auto=?, sentiment_auto=?, sentiment_terms_auto=? WHERE id=?",
                             (topic, label, json.dumps(found.matched_terms, ensure_ascii=False), row["id"]))

            # 3. 存下這次分析的主題摘要（用自動結果，不含人工修正）
            conn.execute(
                "INSERT INTO analysis_runs(batch_id, cluster_count) VALUES(?,?) "
                "ON CONFLICT(batch_id) DO UPDATE SET cluster_count=excluded.cluster_count, created_at=CURRENT_TIMESTAMP",
                (batch_id, len(set(topic_names))))
            run_id = conn.execute("SELECT id FROM analysis_runs WHERE batch_id=?", (batch_id,)).fetchone()["id"]
            conn.execute("DELETE FROM topic_summaries WHERE analysis_run_id=?", (run_id,))
            fresh = conn.execute("SELECT * FROM feedback WHERE batch_id=?", (batch_id,)).fetchall()
            auto_groups: dict[str, list] = {}
            for row in fresh:
                auto_groups.setdefault(row["topic_auto"], []).append(row)
            for index, (name, s) in enumerate(score_groups(auto_groups).items()):
                conn.execute(
                    "INSERT INTO topic_summaries(analysis_run_id, topic_id, label_auto, count, negative_share, "
                    "recent_growth, priority_score, score_components_json, top_terms_json) VALUES(?,?,?,?,?,?,?,?,?)",
                    (run_id, index, name, s["count"], s["negative_share"], s["components"]["growth"], s["score"],
                     json.dumps(s["components"]), json.dumps(top_terms.get(name, []), ensure_ascii=False)))

        source = "星等評分" if any(r["rating"] for r in rows) else "情緒詞典"
        flash(f"分析完成（主題：{'規則分類' if method == 'rules' else '自動分群'}；情緒：{source}）。請抽樣檢視原文。", "success")
        return redirect(url_for("batch", batch_id=batch_id))

    @app.post("/feedback/<int:feedback_id>/labels")
    def labels(feedback_id):
        topic = request.form.get("topic", "").strip() or None
        sentiment = request.form.get("sentiment", "").strip() or None
        if sentiment not in (None, *SENTIMENTS):
            abort(400)
        with db() as conn:
            cur = conn.execute("UPDATE feedback SET topic_manual=?, sentiment_manual=? WHERE id=?",
                               (topic, sentiment, feedback_id))
            if not cur.rowcount:
                abort(404)
            row = conn.execute("SELECT batch_id FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        flash("人工標籤已儲存，儀表板會優先使用人工修正。", "success")
        return redirect(url_for("batch", batch_id=row["batch_id"]))

    # ----- 匯出與報告 -----

    @app.get("/batches/<int:batch_id>/export.csv")
    def export_csv(batch_id):
        _, rows, _ = load_batch(batch_id)
        scores = score_groups(group_by_topic(rows))

        def safe(value) -> str:
            """避免 Excel 把開頭是 = + - @ 的內容當成公式執行。"""
            text = "" if value is None else str(value)
            return "'" + text if text.startswith(("=", "+", "-", "@")) else text

        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["date", "source", "rating", "text", "topic", "sentiment",
                         "topic_auto", "sentiment_auto", "priority_score"])
        for r in sorted(rows, key=lambda r: r["id"]):
            topic = topic_of(r)
            writer.writerow([safe(v) for v in (r["feedback_date"], r["source"], r["rating"], r["body"], topic,
                                               sentiment_of(r), r["topic_auto"], r["sentiment_auto"],
                                               scores.get(topic, {}).get("score", ""))])
        return Response("﻿" + output.getvalue(), mimetype="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename=batch-{batch_id}-feedback.csv"})

    @app.get("/batches/<int:batch_id>/report")
    def report(batch_id):
        b, rows, topics = load_batch(batch_id)
        groups = group_by_topic(rows)
        report_topics = [
            {"label": name, "count": s["count"], "score": s["score"], "components": s["components"],
             "terms": terms_for(topics, name), "examples": groups[name][:3]}
            for name, s in sorted(score_groups(groups).items(), key=lambda kv: kv[1]["score"], reverse=True)]
        return render_template("report.html", b=b, rows=rows, topics=report_topics)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
