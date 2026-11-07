import csv, hashlib, io
from datetime import datetime
from models import FeedbackInput, ImportIssue, ImportResult

def parse_feedback_csv(stream) -> ImportResult:
    raw = stream.read()
    if isinstance(raw, str): raw = raw.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    try: text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc: raise ValueError("檔案必須使用 UTF-8 或 UTF-8 BOM 編碼") from exc
    reader = csv.DictReader(io.StringIO(text))
    required = {"date", "source", "text"}
    missing = required - set(reader.fieldnames or [])
    if missing: raise ValueError("缺少必要欄位：" + "、".join(sorted(missing)))
    rows, issues = [], []
    for row_num, row in enumerate(reader, start=2):
        if row_num > 10001: raise ValueError("單次匯入不得超過 10,000 列")
        body = (row.get("text") or "").strip()
        source = (row.get("source") or "").strip() or "未提供"
        if not body:
            issues.append(ImportIssue(row_num, "text", "回饋內容不可空白")); continue
        raw_date = (row.get("date") or "").strip()
        parsed = None
        if raw_date:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                try: parsed = datetime.strptime(raw_date, fmt).date(); break
                except ValueError: pass
            if parsed is None: issues.append(ImportIssue(row_num, "date", f"日期格式無效：{raw_date}；此筆仍保留但不納入時間趨勢"))
        rating = None
        raw_rating = (row.get("rating") or "").strip()
        if raw_rating:
            if raw_rating.isdigit() and 1 <= int(raw_rating) <= 5:
                rating = int(raw_rating)
            else:
                issues.append(ImportIssue(row_num, "rating", f"評分需為 1–5 的整數：{raw_rating}；此筆改用文字判斷情緒"))
        rows.append(FeedbackInput(parsed, source, body, rating))
        if len(rows) > 10000: raise ValueError("單次匯入有效資料不得超過 10,000 列")
    return ImportResult(digest, rows, issues)
