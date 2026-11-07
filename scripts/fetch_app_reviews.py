"""從 Apple 公開的評論 RSS 抓 App Store 台灣區最近的評論，存成平台可匯入的 CSV。

用法：python scripts/fetch_app_reviews.py <App ID> reviews/輸出.csv
App ID 是 App Store 網址裡 id 後面的數字。RSS 最多提供最近 500 則。
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path


def fetch(app_id: str, pages: int = 10) -> list[dict]:
    reviews = []
    for page in range(1, pages + 1):
        url = f"https://itunes.apple.com/tw/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        data = json.load(urllib.request.urlopen(url, timeout=20))
        entries = data.get("feed", {}).get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]
        entries = [e for e in entries if "im:rating" in e]  # 第一筆可能是 App 本身的資訊
        if not entries:
            break
        for e in entries:
            reviews.append({
                "date": e["updated"]["label"][:10],
                "source": "App Store",
                "rating": e["im:rating"]["label"],
                "text": (e["title"]["label"] + "。" + e["content"]["label"]).replace("\n", " "),
            })
        time.sleep(0.5)  # 放慢一點，避免對伺服器造成負擔
    return reviews


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("用法：python scripts/fetch_app_reviews.py <App ID> reviews/輸出.csv")
    rows = fetch(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source", "rating", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已存 {len(rows)} 則評論到 {out}")
