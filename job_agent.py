import os
import re
import json
import math
import time
import smtplib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any, Tuple

import yaml
import feedparser
from PyPDF2 import PdfReader


SEEN_DB = "seen_jobs.json"


@dataclass
class Job:
    source: str
    title: str
    link: str
    summary: str
    published: str
    published_ts: float


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_cv_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    text = "\n".join(parts)
    return normalize(text)


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def job_id(link: str) -> str:
    return hashlib.sha256(link.encode("utf-8")).hexdigest()[:16]


def load_seen() -> dict:
    if not os.path.exists(SEEN_DB):
        return {"seen": {}}
    with open(SEEN_DB, "r", encoding="utf-8") as f:
        return json.load(f)


def save_seen(db: dict) -> None:
    with open(SEEN_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def fetch_rss(feed_url: str, source_name: str) -> List[Job]:
    feed = feedparser.parse(feed_url)
    jobs: List[Job] = []
    for e in feed.entries:
        title = e.get("title", "").strip()
        link = e.get("link", "").strip()
        summary = (e.get("summary", "") or e.get("description", "") or "").strip()

        # published parsing (best-effort)
        published = e.get("published", "") or e.get("updated", "")
        published_ts = time.time()
        if "published_parsed" in e and e.published_parsed:
            published_ts = time.mktime(e.published_parsed)
        elif "updated_parsed" in e and e.updated_parsed:
            published_ts = time.mktime(e.updated_parsed)

        if title and link:
            jobs.append(Job(
                source=source_name,
                title=title,
                link=link,
                summary=summary,
                published=published,
                published_ts=published_ts
            ))
    return jobs


def compile_regex_list(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def matches_any(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    t = normalize(text)
    hits = 0
    for k in keywords:
        k_norm = normalize(k)
        if k_norm and k_norm in t:
            hits += 1
    return hits


def cosine_sim_bow(a: str, b: str) -> float:
    """
    Simple, dependency-free similarity:
    - bag-of-words tf
    - cosine similarity
    This is not as strong as embeddings, but is reliable and free to run.
    """
    def tf(text: str) -> Dict[str, int]:
        words = re.findall(r"[a-z0-9\-\+]+", normalize(text))
        d: Dict[str, int] = {}
        for w in words:
            if len(w) < 3:
                continue
            d[w] = d.get(w, 0) + 1
        return d

    va = tf(a)
    vb = tf(b)
    if not va or not vb:
        return 0.0

    # dot product
    dot = 0.0
    for w, ca in va.items():
        cb = vb.get(w)
        if cb:
            dot += ca * cb

    # norms
    na = math.sqrt(sum(c * c for c in va.values()))
    nb = math.sqrt(sum(c * c for c in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def freshness_score(published_ts: float, days_boost: int) -> float:
    age_days = max(0.0, (time.time() - published_ts) / 86400.0)
    # Linear decay to zero after days_boost
    if age_days >= days_boost:
        return 0.0
    return 1.0 - (age_days / days_boost)


def score_job(job: Job, cv_text: str, cfg: dict) -> Tuple[float, Dict[str, Any]]:
    mcfg = cfg["matching"]
    scfg = cfg["scoring"]

    title_norm = normalize(job.title)
    blob = f"{job.title}\n{job.summary}"
    blob_norm = normalize(blob)

    include_re = compile_regex_list(mcfg["include_title_regex"])
    exclude_re = compile_regex_list(mcfg["exclude_title_regex"])

    if matches_any(exclude_re, title_norm):
        return -1e9, {"reason": "excluded_title"}

    title_ok = matches_any(include_re, title_norm)
    if not title_ok:
        return -1e9, {"reason": "not_faculty_title"}

    pos_hits = count_keyword_hits(blob_norm, mcfg["positive_keywords"])
    neg_hits = count_keyword_hits(blob_norm, mcfg["negative_keywords"])

    sim = cosine_sim_bow(cv_text, blob_norm)

    fresh = freshness_score(job.published_ts, cfg["search"]["days_freshness_boost"])

    score = 0.0
    score += scfg["title_match"]
    score += pos_hits * scfg["keyword_hit"]
    score += sim * scfg["cv_similarity"]
    score += fresh * scfg["freshness_boost"]
    score += neg_hits * scfg["penalty_negative_keyword"]

    why = []
    if pos_hits:
        why.append(f"Keyword hits: {pos_hits}")
    why.append(f"CV similarity: {sim:.2f}")
    if job.published:
        why.append(f"Posted: {job.published}")
    why.append(f"Source: {job.source}")

    return score, {
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
        "sim": sim,
        "fresh": fresh,
        "why": why
    }


def build_email(jobs: List[Tuple[Job, float, dict]], cfg: dict) -> Tuple[str, str]:
    top_n = cfg["search"]["max_results_email"]
    picked = jobs[:top_n]

    subject = f"Faculty job matches (EU/USA) — {datetime.now().strftime('%Y-%m-%d')}"

    lines = []
    lines.append("Top faculty matches (ranked):")
    lines.append("")
    for i, (j, s, meta) in enumerate(picked, 1):
        lines.append(f"{i}. {j.title}")
        lines.append(f"   Link: {j.link}")
        if j.published:
            lines.append(f"   Posted: {j.published}")
        lines.append(f"   Score: {s:.2f}")
        lines.append(f"   Why: " + "; ".join(meta.get("why", [])))
        lines.append("")

    if not picked:
        lines.append("No strong faculty matches found from the configured RSS sources today.")
        lines.append("Tip: add more discipline/location feeds in config.yaml to widen coverage.")

    body_text = "\n".join(lines)
    body_html = "<pre style='font-family: ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap;'>" \
                + body_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") \
                + "</pre>"
    return subject, body_html


def send_email_smtp(cfg: dict, subject: str, html: str) -> None:
    ecfg = cfg["email"]

    password = os.environ.get("SMTP_PASSWORD", "")
    if not password:
        raise RuntimeError("Missing SMTP_PASSWORD env var (use a Gmail App Password).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = ecfg["from_email"]
    msg["To"] = ecfg["to_email"]
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(ecfg["smtp_host"], ecfg["smtp_port"]) as server:
        server.starttls()
        server.login(ecfg["smtp_username"], password)
        server.sendmail(ecfg["from_email"], [ecfg["to_email"]], msg.as_string())


def main():
    cfg = load_config("config.yaml")

    cv_pdf = cfg["cv"]["pdf_path"]
    if not os.path.exists(cv_pdf):
        raise FileNotFoundError(f"CV PDF not found at '{cv_pdf}'. Put your CV file there or update config.yaml.")

    cv_text = read_cv_text(cv_pdf)

    # Fetch jobs
    collected: List[Job] = []

    for url in cfg["sources"]["higheredjobs_rss"]:
        if url.endswith("/rss/"):
            # directory page is not a feed; skip safely
            continue
        collected.extend(fetch_rss(url, "HigherEdJobs"))

    for url in cfg["sources"]["jobs_ac_uk_rss"]:
        collected.extend(fetch_rss(url, "jobs.ac.uk"))

    # Deduplicate by link
    by_link: Dict[str, Job] = {}
    for j in collected:
        by_link[j.link] = j
    deduped = list(by_link.values())

    # Load seen DB
    db = load_seen()
    seen = db.get("seen", {})

    scored: List[Tuple[Job, float, dict]] = []
    for j in deduped:
        jid = job_id(j.link)
        if jid in seen:
            continue
        s, meta = score_job(j, cv_text, cfg)
        if s > -1e8:
            scored.append((j, s, meta))

        # Mark as seen regardless (prevents repeated evaluation loops)
        seen[jid] = {"link": j.link, "title": j.title, "ts": time.time()}

    db["seen"] = seen
    save_seen(db)

    scored.sort(key=lambda x: x[1], reverse=True)

    subject, html = build_email(scored, cfg)

    # Only email if there are matches OR you want daily "no results" notifications
    send_email_smtp(cfg, subject, html)


if __name__ == "__main__":
    main()
