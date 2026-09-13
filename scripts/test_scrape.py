"""
Quick verification script to test HTML-to-Markdown parsing on a single PyTorch doc page.

Usage:
    python scripts/test_scrape.py [URL]
"""

import sys
from pathlib import Path

# Add scripts directory to sys.path for direct invocation
sys.path.insert(0, str(Path(__file__).parent))

from scrape_docs import fetch_page, extract_main_content

DEFAULT_URL = "https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f"Testing scraper on: {url}")

    try:
        html, final_url = fetch_page(url)
        print(f"Resolved URL: {final_url}")

        title, markdown = extract_main_content(html)
        print(f"\nTitle: {title}")
        print("=" * 50)
        print("Markdown Preview (first 1000 chars):")
        print("=" * 50)
        print(markdown[:1000])
        print("=" * 50)
        print(f"Total Markdown Length: {len(markdown)} characters")
    except Exception as e:
        print(f"Error during scraping test: {e}")


if __name__ == "__main__":
    main()
