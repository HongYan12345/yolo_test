from __future__ import annotations

import html
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


HEADERS = {
    "User-Agent": "Mozilla/5.0 MapleStoryAI crawler probe",
    "Accept": "text/html,image/png,image/webp,image/gif,image/jpeg,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def fetch(url: str, limit: int | None = None) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read(limit)
        return resp.status, resp.headers.get("content-type", ""), data


def text_fetch(url: str) -> str:
    status, content_type, data = fetch(url)
    print(f"FETCH {url}")
    print(f"  status={status} content_type={content_type} bytes={len(data)}")
    return data.decode("utf-8", "ignore")


def absolutize(base_url: str, value: str) -> str:
    value = html.unescape(value.strip())
    return urllib.parse.urljoin(base_url, value)


def extract_attrs(page_url: str, text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"""(?P<attr>src|data-src|href|data-original)\s*=\s*["'](?P<url>.*?)["']""", text, re.I | re.S):
        attr = m.group("attr").lower()
        url = absolutize(page_url, m.group("url"))
        out.append((attr, url))
    return out


def print_candidates(page_url: str, text: str) -> None:
    title = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
    print("TITLE:", html.unescape(re.sub(r"\s+", " ", title.group(1))).strip() if title else "")

    attrs = extract_attrs(page_url, text)

    print("\nIMAGE / ASSET candidates:")
    seen = set()
    for attr, url in attrs:
        low = url.lower()
        if any(k in low for k in [".png", ".webp", ".gif", ".jpg", ".jpeg", "monster", "item", "icon", "images", "upload", "wz"]):
            if url not in seen:
                seen.add(url)
                print(f"  {attr}: {url}")

    print("\nMONSTER links:")
    seen.clear()
    for attr, url in attrs:
        if "/monster/" in url and url not in seen:
            seen.add(url)
            print(f"  {url}")

    print("\nITEM-like links:")
    seen.clear()
    for attr, url in attrs:
        low = url.lower()
        if any(k in low for k in ["/item/", "/equip/", "/drop/", "/prop/", "/cash/", "/consume/", "/etc/"]) and url not in seen:
            seen.add(url)
            print(f"  {url}")

    print("\nID-like snippets:")
    for pattern in [
        r"/monster/(\d+)",
        r"/item/(\d+)",
        r"/equip/(\d+)",
        r"monster[_/-](\d+)",
        r"item[_/-](\d+)",
        r"(\d{5,8})\.(?:png|webp|gif)",
    ]:
        ids = sorted(set(re.findall(pattern, text, re.I)))
        if ids:
            print(f"  {pattern}: {ids[:30]}")


def main() -> int:
    urls = sys.argv[1:] or [
        "https://mxdzlk.com/monster/100100",
        "https://mxdzlk.com/monster",
        "https://mxdzlk.com/items",
        "https://mxdzlk.com/item",
        "https://mxdzlk.com/equip",
    ]

    out_dir = Path("tmp_probe")
    out_dir.mkdir(exist_ok=True)

    for url in urls:
        print("\n" + "=" * 100)
        print(url)
        try:
            text = text_fetch(url)
            safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", urllib.parse.urlparse(url).path.strip("/") or "root")
            (out_dir / f"{safe_name}.html").write_text(text, encoding="utf-8")
            print_candidates(url, text)
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
