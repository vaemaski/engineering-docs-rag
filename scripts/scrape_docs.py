"""
Scrapes a BOUNDED set of PyTorch doc pages (not a crawl).
Saves each page as clean markdown + a metadata JSON sidecar.

Usage:
    python scripts/scrape_docs.py
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString

# ---- Bounded URL list: torch.utils.data + torch.nn basics ----
# Add/remove URLs here to control corpus size. Keep this list explicit
# and small on purpose -- this is a curated slice, not a full crawl.
DOC_URLS = [
    # torch.utils.data overview
    "https://docs.pytorch.org/docs/stable/data.html",
    # torch.nn overview + core layer docs
    "https://docs.pytorch.org/docs/stable/nn.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html",
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html",
]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def slugify(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace(".html", "") or "index"


def fetch_page(url: str, max_redirects: int = 3) -> tuple[str, str]:
    """
    Fetches URL handling both HTTP redirects and HTML <meta http-equiv="refresh"> redirects used by PyTorch docs.
    Returns (html_content, final_url).
    """
    current_url = url
    for _ in range(max_redirects):
        resp = requests.get(current_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
        if meta_refresh:
            content = meta_refresh.get("content", "")
            if "url=" in content:
                target = content.split("url=")[-1].strip()
                current_url = urljoin(current_url, target)
                continue
        return resp.text, current_url
    return resp.text, current_url


def _element_to_md(element) -> str:
    lines = []
    for child in element.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                lines.append(text)
            continue
        
        name = child.name
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            h_level = int(name[1])
            text = child.get_text(" ", strip=True).replace("¶", "")
            if text:
                lines.append(f"\n\n{'#' * h_level} {text}\n")
        elif name == "p":
            text = child.get_text(" ", strip=True).replace("¶", "")
            if text:
                lines.append(f"\n\n{text}")
        elif name == "pre":
            text = child.get_text().strip()
            if text:
                lines.append(f"\n\n```python\n{text}\n```\n")
        elif name in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                t = li.get_text(" ", strip=True).replace("¶", "")
                if t:
                    lines.append(f"\n- {t}")
        elif name == "dl":
            lines.append(_element_to_md(child))
        elif name == "dt":
            t = child.get_text(" ", strip=True).replace("¶", "")
            if t:
                lines.append(f"\n\n**{t}**")
        elif name == "dd":
            t = _element_to_md(child)
            if t:
                lines.append(f"\n{t}")
        elif name in ("section", "div", "article"):
            lines.append(_element_to_md(child))
            
    return "".join(lines)


def extract_main_content(html: str) -> tuple[str, str]:
    """Returns (title, clean_text_markdown) from a PyTorch docs page."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True).replace("¶", "") if title_tag else "Untitled"

    # PyTorch docs main content lives in <article> or role=main
    main = soup.find("article") or soup.find(attrs={"role": "main"}) or soup.find("main")
    if main is None:
        main = soup.body if soup.body else soup

    # Strip nav/scripts/styles/asides/footers/headers
    for tag in main.find_all(["nav", "script", "style", "aside", "footer", "header"]):
        tag.decompose()

    # Remove Sphinx permalinks (e.g. ¶)
    for anchor in main.find_all("a", class_="headerlink"):
        anchor.decompose()

    markdown_body = _element_to_md(main)
    return title, markdown_body.strip()


def scrape():
    manifest = []
    for url in DOC_URLS:
        slug = slugify(url)
        out_path = RAW_DIR / f"{slug}.md"
        meta_path = RAW_DIR / f"{slug}.meta.json"

        print(f"Fetching {url} ...")
        try:
            html, final_url = fetch_page(url)
        except requests.RequestException as e:
            print(f"  FAILED: {e}")
            continue

        title, content = extract_main_content(html)

        out_path.write_text(f"# {title}\n\nSource: {final_url}\n\n{content}", encoding="utf-8")
        meta = {"url": url, "final_url": final_url, "title": title, "slug": slug}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append(meta)

        time.sleep(1)  # be polite to the docs server

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Scraped {len(manifest)}/{len(DOC_URLS)} pages into {RAW_DIR}")


if __name__ == "__main__":
    scrape()