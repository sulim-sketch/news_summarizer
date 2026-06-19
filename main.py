"""
Yahoo Finance Stock Market News 자동 브리핑 파이프라인
=====================================================
소스: https://finance.yahoo.com/topic/stock-market-news/{page}/
대상: page 1~3
목적: 2단계 필터링 후 통과한 기사를 Gemini API로 요약 생성
"""

import asyncio
import re
import os
from datetime import date
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

BASE_URL = "https://finance.yahoo.com/topic/stock-market-news"
PAGES = [1, 2, 3]

# ─────────────────────────────────────────
# 필터 키워드
# ─────────────────────────────────────────

EXCLUDE_TITLE_KEYWORDS = [
    "should you buy", "is it too late", "top stocks to buy",
    "best stocks", "time to buy", "time to sell",
    "why i", "here's why", "you should know",
    "worth buying", "worth watching", "stocks to buy now",
    "never sell", "hold forever", "skyrocket",
    "3 reasons to", "set it and forget",
]

EVENT_KEYWORDS = [
    # M&A
    "acqui", "merger", "takeover", "deal", "buys", "purchase",
    # 실적/가이던스
    "earnings", "revenue", "guidance", "beats", "misses", "eps",
    "quarterly results", "q1", "q2", "q3", "q4", "profit",
    # 규제/정책
    "fda", "sec", "ftc", "approved", "rejected", "ban",
    "fine", "penalty", "chips act", "subsidy", "tariff",
    "sanction", "export control", "restriction",
    # 파트너십/기술 마일스톤
    "launch", "production", "partnership", "contract",
    "agreement", "collaboration", "signs", "awarded",
    # 인사
    "ceo", "cfo", "appoints", "resigns", "fired", "names",
    "steps down", "stepping down",
    # 지정학
    "restrict", "blocks", "halts", "war", "iran",
]


# ─────────────────────────────────────────
# STEP 1: 목록 페이지에서 기사 제목 + URL + 티커 수집
# ─────────────────────────────────────────

async def fetch_article_list(page, page_num: int) -> list[dict]:
    """
    page=1: https://finance.yahoo.com/topic/stock-market-news/
    page=2이상: https://finance.yahoo.com/topic/stock-market-news/{page_num}/

    HTML 구조 차이:
    - page=1: JS 렌더링 후 <h2>/<h3> 태그에 기사 제목, 하단에 티커 링크 포함
    - page=2이상: 동일한 레이아웃이지만 URL 경로에 숫자가 붙음.
      일부 브라우저 환경에서 초기 로드 시 빈 컨테이너가 반환될 수 있으므로
      networkidle 대신 특정 선택자가 나타날 때까지 명시적으로 대기 필요.
    """
    if page_num == 1:
        url = f"{BASE_URL}/"
    else:
        url = f"{BASE_URL}/{page_num}/"

    await page.goto(url, wait_until="networkidle", timeout=30000)

    # page=2 이상에서 컨텐츠 로딩이 늦는 경우 추가 대기
    if page_num >= 2:
        try:
            await page.wait_for_selector("h3 a, h2 a", timeout=10000)
        except Exception:
            pass  # 타임아웃 시 현재 상태로 진행

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    articles = []

    # 기사 카드 탐색: Yahoo Finance는 <h2>/<h3> 안에 <a>로 제목+링크 구성
    for heading in soup.find_all(["h2", "h3"]):
        a_tag = heading.find("a", href=True)
        if not a_tag:
            continue

        title = a_tag.get_text(strip=True)
        href = a_tag["href"]

        # 상대경로 처리
        if href.startswith("/"):
            article_url = "https://finance.yahoo.com" + href
        elif href.startswith("http"):
            article_url = href
        else:
            continue

        articles.append({
            "title": title,
            "url": article_url,
        })

    return articles


# ─────────────────────────────────────────
# STEP 2-1: 1단계 필터 — 의견성 기사 제목 제외 (본문 fetch 전)
# ─────────────────────────────────────────

def passes_title_filter(title: str) -> tuple[bool, str]:
    title_lower = title.lower()
    for kw in EXCLUDE_TITLE_KEYWORDS:
        if kw in title_lower:
            return False, f"의견성 제목 ({kw})"
    return True, ""


# ─────────────────────────────────────────
# STEP 2-2: 2단계 필터 — 이벤트 드리븐 확인
# ─────────────────────────────────────────

def passes_event_filter(title: str, body: str) -> tuple[bool, str]:
    full_text = (title + " " + body[:3000]).lower()
    matched = [kw for kw in EVENT_KEYWORDS if kw in full_text]
    if matched:
        return True, f"이벤트: {matched[:3]}"
    return False, "이벤트 키워드 없음"


# ─────────────────────────────────────────
# STEP 3: 기사 본문 fetch (Playwright)
# ─────────────────────────────────────────

async def fetch_article_body(page, url: str) -> str:
    """본문 텍스트 + 본문 내 티커 반환"""
    try:
        await page.goto(url, wait_until="networkidle", timeout=20000)
        html = await page.content()
    except Exception:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    article = soup.find("article") or soup
    body_text = article.get_text(separator="\n", strip=True)

    return body_text[:5000]


# ─────────────────────────────────────────
# STEP 4: Gemini로 요약 생성
# ─────────────────────────────────────────

SUMMARY_PROMPT = """
아래 기사를 읽고 다음 형식으로 보고용 자료를 작성해줘.
대표/부대표에게 보고하는 자료이므로 전문 용어는 쉽게 풀어서 설명하고, 내용은 간결하게 정리해줘.

작성 형식:
### 종목명 (티커)
한 줄 요약 (주요 이벤트 및 주가 등락률)

* 내용 1
* 내용 2
* *[요약 텍스트 (날짜)](URL)*

작성 규칙:
- 수치(매출, EPS, 주가 등락률 등) 반드시 포함
- 불렛 포인트 최대 4개
- Source 항목은 요약 텍스트 자체를 마크다운 하이퍼링크로 만들 것: * *[요약 텍스트 (날짜)](URL)*
- 별도로 URL을 들여쓰기해서 보여주지 말 것
- 사전 학습 지식 활용 금지, 기사 내용만 사용
- 장중 작성 기사의 경우 정규장 종료 후 수익률로 변경

기사 URL: {url}

기사 내용:
{body}
"""

async def summarize_article(url: str, body: str) -> str:
    prompt = SUMMARY_PROMPT.format(url=url, body=body)
    response = model.generate_content(prompt)
    return response.text


# ─────────────────────────────────────────
# STEP 5: 중요도 정렬
# ─────────────────────────────────────────

RANKING_PROMPT = """
아래 종목 요약들을 투자 보고용으로 중요도 순서로 재배열해줘.

중요도 기준 (순서대로 적용):
1. 시장 임팩트 규모 (관련 기업 시가총액, 거래 규모)
2. 이벤트의 신규성 (M&A > 실적 > 파트너십 > 규제)
3. 주가 등락 폭

요약들:
{summaries}
"""

async def rank_summaries(summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(summaries)
    prompt = RANKING_PROMPT.format(summaries=combined)
    response = model.generate_content(prompt)
    return response.text


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────

async def main():
    all_articles = []
    passed_articles = []
    summaries = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        list_page = await browser.new_page()
        article_page = await browser.new_page()

        # ── 기사 목록 수집 (page 1~3) ──
        for page_num in PAGES:
            print(f"[Page {page_num}] 기사 목록 수집 중...")
            articles = await fetch_article_list(list_page, page_num)
            all_articles.extend(articles)
            print(f"  → {len(articles)}건 수집")

        print(f"\n총 {len(all_articles)}건 수집\n{'='*50}")

        # ── 2단계 필터링 + 요약 ──
        for art in all_articles:
            title = art["title"]
            url = art["url"]

            # 1단계: 제목 필터 (본문 fetch 전)
            ok, reason = passes_title_filter(title)
            if not ok:
                print(f"[1단계 제외] {title[:50]} → {reason}")
                continue

            # 본문 fetch
            body = await fetch_article_body(article_page, url)

            # 2단계: 이벤트 필터
            ok, reason = passes_event_filter(title, body)
            if not ok:
                print(f"[2단계 제외] {title[:50]} → {reason}")
                continue

            print(f"[통과] {title[:50]}")
            passed_articles.append(art)

            # 요약 생성
            summary = await summarize_article(url, body)
            summaries.append(summary)

        await browser.close()

    print(f"\n{'='*50}")
    print(f"필터 통과: {len(passed_articles)}건 / 전체: {len(all_articles)}건\n")

    # 중요도 정렬
    if summaries:
        final_output = await rank_summaries(summaries)
        print(final_output)

        # 파일 저장
        output_file = f"briefing_{date.today()}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"\n저장 완료: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
