from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BASE_URL = "https://mxdzlk.com"
STATIC_BASE_URL = "https://static.mapleartale.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 MapleStoryAI asset crawler (+local research)",
    "Accept": "text/html,image/png,image/webp,image/gif,image/jpeg,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": BASE_URL + "/",
}

IMAGE_EXT_BY_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"\xff\xd8\xff": ".jpg",
    b"RIFF": ".webp",
}

METADATA_FIELDS = [
    "id",
    "kind",
    "name",
    "item_type",
    "source",
    "page_url",
    "image_url",
    "file_path",
    "status",
    "http_status",
    "content_type",
    "size_bytes",
    "sha256",
    "error",
    "downloaded_at",
    "discovered_from",
]


@dataclass
class AssetRecord:
    id: str
    kind: str
    name: str
    item_type: str
    source: str
    page_url: str
    image_url: str
    file_path: str = ""
    status: str = "discovered"
    http_status: str = ""
    content_type: str = ""
    size_bytes: str = ""
    sha256: str = ""
    error: str = ""
    downloaded_at: str = ""
    discovered_from: str = ""


@dataclass(frozen=True)
class ListSource:
    name: str
    kind: str
    url: str
    item_type: str = ""


class RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._last_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return

        with self._lock:
            now = time.monotonic()
            wait_seconds = self.delay_seconds - (now - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(message, flush=True)


def build_page_url(source_url: str, page_num: int) -> str:
    if page_num <= 1:
        return source_url

    parsed = urllib.parse.urlparse(source_url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = {k: v for k, v in query_pairs}
    query["page_num"] = str(page_num)

    rebuilt = parsed._replace(query=urllib.parse.urlencode(query))
    return urllib.parse.urlunparse(rebuilt)


def request_bytes(
    url: str,
    limiter: RateLimiter,
    *,
    timeout: int,
    retries: int,
    referer: str | None = None,
) -> tuple[int, str, bytes]:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        limiter.wait()
        headers = dict(HEADERS)
        if referer:
            headers["Referer"] = referer

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.headers.get("content-type", ""), resp.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            # 404/410 are deterministic missing assets; retrying them wastes a lot of time during full crawls.
            if exc.code in {404, 410}:
                raise

            # 403/429/5xx may recover after waiting.
            if attempt < retries:
                sleep_seconds = min(30.0, 2.0 ** attempt)
                log(f"  retry after HTTP {exc.code}: {url} ({sleep_seconds:.1f}s) reason={exc.reason}")
                time.sleep(sleep_seconds)
            else:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                sleep_seconds = min(30.0, 2.0 ** attempt)
                log(f"  retry after {type(exc).__name__}: {url} ({sleep_seconds:.1f}s) error={exc}")
                time.sleep(sleep_seconds)
            else:
                raise

    raise RuntimeError(f"request failed: {url}: {last_error}")


def request_text(url: str, limiter: RateLimiter, *, timeout: int, retries: int) -> tuple[int, str, str]:
    status, content_type, data = request_bytes(url, limiter, timeout=timeout, retries=retries)
    return status, content_type, data.decode("utf-8", "ignore")


def strip_tags(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", "", value, flags=re.I | re.S)
    value = re.sub(r"<style\b.*?</style>", "", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_title_name(text: str) -> str:
    match = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
    if not match:
        return ""
    title = strip_tags(match.group(1))
    # Examples:
    #   蜗牛 - 怪物详情 - 怀旧冒险岛资料库
    #   红药水 - 道具详情 - 怀旧冒险岛资料库
    return title.split(" - ", 1)[0].strip()


def parse_max_page(text: str) -> int:
    page_nums = [int(x) for x in re.findall(r"[?&]page_num=(\d+)", text)]
    return max(page_nums) if page_nums else 1


def extract_attr(tag: str, attr_name: str) -> str:
    match = re.search(
        rf"""\b{re.escape(attr_name)}\s*=\s*["'](.*?)["']""",
        tag,
        re.I | re.S,
    )
    return html.unescape(match.group(1)).strip() if match else ""


def guess_name_near_image(text: str, image_url: str, asset_id: str) -> str:
    # Try img alt/title first.
    escaped = re.escape(image_url)
    for pattern in [
        rf"<img\b[^>]*src\s*=\s*['\"]{escaped}['\"][^>]*>",
        rf"<img\b[^>]*(?:mob|item)/{re.escape(asset_id)}\.png[^>]*>",
    ]:
        match = re.search(pattern, text, re.I | re.S)
        if match:
            tag = match.group(0)
            return extract_attr(tag, "alt") or extract_attr(tag, "title")

    # Then inspect a small surrounding HTML snippet.
    idx = text.find(image_url)
    if idx < 0:
        idx = text.find(f"/{asset_id}.png")
    if idx >= 0:
        snippet = text[max(0, idx - 500) : min(len(text), idx + 500)]
        candidate = strip_tags(snippet)
        candidate = re.sub(rf"\b{re.escape(asset_id)}\b", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        # Avoid returning a huge menu/pagination text as name.
        if 0 < len(candidate) <= 40:
            return candidate

    return ""


def discover_assets_from_page(
    *,
    source: ListSource,
    page_url: str,
    text: str,
) -> list[AssetRecord]:
    records: dict[tuple[str, str], AssetRecord] = {}

    if source.kind == "monster":
        image_pattern = r"(?:(?:https?:)?//)static\.mapleartale\.com/images/mob/(\d+)\.(png|webp|gif|jpg|jpeg)"
        detail_path = "monster"
        image_dir = "mob"
    else:
        image_pattern = r"(?:(?:https?:)?//)static\.mapleartale\.com/images/item/(\d+)\.(png|webp|gif|jpg|jpeg)"
        detail_path = "item"
        image_dir = "item"

    for match in re.finditer(image_pattern, text, re.I):
        asset_id = match.group(1)
        image_url = f"{STATIC_BASE_URL}/images/{image_dir}/{asset_id}.{match.group(2).lower()}"
        page_match = re.search(
            rf"https?://mxdzlk\.com/{detail_path}/{re.escape(asset_id)}/?",
            text,
            re.I,
        )
        asset_page_url = page_match.group(0) if page_match else f"{BASE_URL}/{detail_path}/{asset_id}/"

        key = (source.kind, asset_id)
        if key not in records:
            records[key] = AssetRecord(
                id=asset_id,
                kind=source.kind,
                name=guess_name_near_image(text, image_url, asset_id),
                item_type=source.item_type,
                source=source.name,
                page_url=asset_page_url,
                image_url=image_url,
                discovered_from=page_url,
            )

    return list(records.values())


def load_existing_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            asset_id = row.get("id", "")
            if asset_id:
                rows[asset_id] = {field: row.get(field, "") for field in METADATA_FIELDS}
    return rows


def load_records_from_metadata(output_root: Path, kind: str) -> list[AssetRecord]:
    """Load previously discovered records and continue downloading without re-scanning list pages."""
    metadata_path = output_root / ("monsters" if kind == "monster" else "items") / "metadata.csv"
    rows = load_existing_metadata(metadata_path)
    records: list[AssetRecord] = []

    for row in rows.values():
        image_url = row.get("image_url", "")
        asset_id = row.get("id", "")
        if not asset_id or not image_url:
            continue

        record_kind = row.get("kind") or kind
        if record_kind != kind:
            continue

        records.append(
            AssetRecord(
                id=asset_id,
                kind=record_kind,
                name=row.get("name", ""),
                item_type=row.get("item_type", ""),
                source=row.get("source", ""),
                page_url=row.get("page_url", ""),
                image_url=image_url,
                file_path=row.get("file_path", ""),
                status=row.get("status", "discovered"),
                http_status=row.get("http_status", ""),
                content_type=row.get("content_type", ""),
                size_bytes=row.get("size_bytes", ""),
                sha256=row.get("sha256", ""),
                error=row.get("error", ""),
                downloaded_at=row.get("downloaded_at", ""),
                discovered_from=row.get("discovered_from", ""),
            )
        )

    return records


def write_metadata(path: Path, rows_by_id: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for asset_id in sorted(rows_by_id, key=lambda x: int(x) if x.isdigit() else x):
            row = {field: rows_by_id[asset_id].get(field, "") for field in METADATA_FIELDS}
            writer.writerow(row)


def merge_metadata(output_root: Path, records: Iterable[AssetRecord]) -> None:
    grouped: dict[str, list[AssetRecord]] = {"monster": [], "item": []}
    for record in records:
        grouped.setdefault(record.kind, []).append(record)

    for kind, kind_records in grouped.items():
        if not kind_records:
            continue

        metadata_path = output_root / ("monsters" if kind == "monster" else "items") / "metadata.csv"
        rows = load_existing_metadata(metadata_path)
        for record in kind_records:
            current = rows.get(record.id, {})
            new_row = asdict(record)
            # Preserve successful existing downloads if this run only rediscovered the asset.
            if current.get("status") in {"downloaded", "exists"} and record.status == "discovered":
                merged = dict(current)
                for key in ["name", "item_type", "source", "page_url", "image_url", "discovered_from"]:
                    if new_row.get(key) and not merged.get(key):
                        merged[key] = new_row[key]
                rows[record.id] = merged
            else:
                rows[record.id] = new_row

        write_metadata(metadata_path, rows)


def get_image_extension(data: bytes, image_url: str) -> str:
    lower_path = urllib.parse.urlparse(image_url).path.lower()
    suffix = Path(lower_path).suffix
    if suffix in {".png", ".webp", ".gif", ".jpg", ".jpeg"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    for magic, ext in IMAGE_EXT_BY_MAGIC.items():
        if data.startswith(magic):
            return ext
    return ".png"


def is_probably_image(data: bytes, content_type: str) -> bool:
    low = content_type.lower()
    if low.startswith("image/"):
        return True
    return any(data.startswith(magic) for magic in IMAGE_EXT_BY_MAGIC)


def download_asset(
    *,
    record: AssetRecord,
    output_root: Path,
    limiter: RateLimiter,
    timeout: int,
    retries: int,
    force: bool,
) -> AssetRecord:
    kind_dir = output_root / ("monsters" if record.kind == "monster" else "items") / "raw"
    kind_dir.mkdir(parents=True, exist_ok=True)

    default_path = kind_dir / f"{record.id}.png"
    existing_files = list(kind_dir.glob(f"{record.id}.*"))

    if existing_files and not force:
        file_path = existing_files[0]
        data = file_path.read_bytes()
        record.file_path = file_path.as_posix()
        record.status = "exists"
        record.size_bytes = str(len(data))
        record.sha256 = hashlib.sha256(data).hexdigest()
        record.downloaded_at = now_iso()
        return record

    try:
        status, content_type, data = request_bytes(
            record.image_url,
            limiter,
            timeout=timeout,
            retries=retries,
            referer=record.page_url or BASE_URL + "/",
        )

        record.http_status = str(status)
        record.content_type = content_type

        if not is_probably_image(data, content_type):
            raise ValueError(f"response is not an image: content_type={content_type!r} bytes={len(data)}")

        ext = get_image_extension(data, record.image_url)
        file_path = default_path.with_suffix(ext)
        file_path.write_bytes(data)

        record.file_path = file_path.as_posix()
        record.status = "downloaded"
        record.size_bytes = str(len(data))
        record.sha256 = hashlib.sha256(data).hexdigest()
        record.error = ""
        record.downloaded_at = now_iso()
    except urllib.error.HTTPError as exc:
        record.http_status = str(exc.code)
        record.status = "missing" if exc.code in {404, 410} else "failed"
        record.error = f"HTTPError: HTTP Error {exc.code}: {exc.reason}"
        record.downloaded_at = now_iso()
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        record.downloaded_at = now_iso()

    return record


def get_sources(args: argparse.Namespace) -> list[ListSource]:
    sources: list[ListSource] = []

    if args.kind in {"monster", "all"}:
        sources.append(ListSource(name="mxdzlk_monster", kind="monster", url=BASE_URL + "/monster/"))

    if args.kind in {"item", "all"}:
        selected_item_sources = set(args.item_sources)
        item_source_map = {
            "item": ListSource(name="mxdzlk_item", kind="item", url=BASE_URL + "/item/", item_type="item"),
            "equip": ListSource(name="mxdzlk_item_equip", kind="item", url=BASE_URL + "/item/equip/", item_type="equip"),
            "type2": ListSource(name="mxdzlk_item_type_2", kind="item", url=BASE_URL + "/item/?item_type=2", item_type="item_type_2"),
            "type3": ListSource(name="mxdzlk_item_type_3", kind="item", url=BASE_URL + "/item/?item_type=3", item_type="item_type_3"),
            "type4": ListSource(name="mxdzlk_item_type_4", kind="item", url=BASE_URL + "/item/?item_type=4", item_type="item_type_4"),
            "type5": ListSource(name="mxdzlk_item_type_5", kind="item", url=BASE_URL + "/item/?item_type=5", item_type="item_type_5"),
        }
        for key in args.item_sources_order:
            if key in selected_item_sources:
                sources.append(item_source_map[key])

    return sources


def discover_from_source(
    *,
    source: ListSource,
    limiter: RateLimiter,
    timeout: int,
    retries: int,
    max_pages: int,
    max_assets: int,
) -> list[AssetRecord]:
    records_by_key: dict[tuple[str, str], AssetRecord] = {}
    total_pages: int | None = None
    page_num = 1

    while True:
        page_url = build_page_url(source.url, page_num)
        log(f"[DISCOVER] {source.name} page={page_num} url={page_url}")

        try:
            status, content_type, text = request_text(page_url, limiter, timeout=timeout, retries=retries)
            log(f"  status={status} content_type={content_type} chars={len(text)}")
        except urllib.error.HTTPError as exc:
            log(f"  HTTP {exc.code}; stop source {source.name}")
            break
        except Exception as exc:
            log(f"  ERROR {type(exc).__name__}: {exc}; stop source {source.name}")
            break

        if total_pages is None:
            discovered_max_page = parse_max_page(text)
            total_pages = discovered_max_page if max_pages <= 0 else min(discovered_max_page, max_pages)
            log(f"  source pages: discovered={discovered_max_page}, will_scan={total_pages}")

        page_records = discover_assets_from_page(source=source, page_url=page_url, text=text)
        log(f"  assets_on_page={len(page_records)}")

        for record in page_records:
            key = (record.kind, record.id)
            if key not in records_by_key:
                records_by_key[key] = record
                if max_assets > 0 and len(records_by_key) >= max_assets:
                    log(f"  reached max_assets={max_assets} for source {source.name}")
                    return list(records_by_key.values())

        if total_pages is None or page_num >= total_pages:
            break

        page_num += 1

    return list(records_by_key.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl mxdzlk monster/item image assets for MapleStoryAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kind", choices=["monster", "item", "all"], default="monster", help="resource kind to crawl")
    parser.add_argument(
        "--output",
        default="assets",
        help="output directory. Images will be saved under assets/monsters/raw and assets/items/raw",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="pages to scan per list source. Use 0 to scan all discovered pages.",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=0,
        help="max assets per list source. Use 0 for no per-source limit.",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between HTTP requests")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=2, help="retry count for failed HTTP requests")
    parser.add_argument("--workers", type=int, default=1, help="parallel download workers. Use 1 for sequential downloads")
    parser.add_argument("--dry-run", action="store_true", help="only discover assets and write metadata, do not download images")
    parser.add_argument("--force", action="store_true", help="re-download images even if local files exist")
    parser.add_argument(
        "--item-sources",
        default="item,equip,type2,type3,type4,type5",
        help="comma-separated item sources: item,equip,type2,type3,type4,type5",
    )
    parser.add_argument(
        "--metadata-interval",
        type=int,
        default=1,
        help="write metadata every N downloaded records. Use 0 to only write metadata at the end",
    )
    parser.add_argument(
        "--no-write-metadata",
        action="store_true",
        help="do not write metadata.csv files",
    )
    parser.add_argument(
        "--from-metadata",
        action="store_true",
        help="skip list-page discovery and download records already present in metadata.csv",
    )

    args = parser.parse_args()
    valid_item_sources = ["item", "equip", "type2", "type3", "type4", "type5"]
    args.item_sources_order = valid_item_sources
    args.item_sources = [x.strip() for x in args.item_sources.split(",") if x.strip()]

    invalid = sorted(set(args.item_sources) - set(valid_item_sources))
    if invalid:
        parser.error(f"invalid --item-sources value(s): {', '.join(invalid)}")

    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output)
    limiter = RateLimiter(args.delay)

    sources = get_sources(args)
    if not sources:
        log("No sources selected.")
        return 1

    all_records: list[AssetRecord] = []

    log("=== mxdzlk asset crawler ===")
    log(f"kind={args.kind} output={output_root} max_pages={args.max_pages} max_assets={args.max_assets} dry_run={args.dry_run} from_metadata={args.from_metadata}")

    if args.from_metadata:
        if args.kind == "all":
            all_records.extend(load_records_from_metadata(output_root, "monster"))
            all_records.extend(load_records_from_metadata(output_root, "item"))
        else:
            all_records.extend(load_records_from_metadata(output_root, args.kind))
        log(f"[METADATA LOAD] records={len(all_records)}")
    else:
        log("sources=" + ", ".join(source.name for source in sources))

        for source in sources:
            records = discover_from_source(
                source=source,
                limiter=limiter,
                timeout=args.timeout,
                retries=args.retries,
                max_pages=args.max_pages,
                max_assets=args.max_assets,
            )
            log(f"[SOURCE DONE] {source.name}: discovered_unique={len(records)}")
            all_records.extend(records)

        # Global de-duplication by kind + id. Prefer the first source that found the asset.
        unique_records: dict[tuple[str, str], AssetRecord] = {}
        for record in all_records:
            unique_records.setdefault((record.kind, record.id), record)
        all_records = list(unique_records.values())

        log(f"[DISCOVERY DONE] unique_total={len(all_records)} monster={sum(r.kind == 'monster' for r in all_records)} item={sum(r.kind == 'item' for r in all_records)}")

        if not args.no_write_metadata:
            merge_metadata(output_root, all_records)
            log("[METADATA] wrote discovered records")

    if not args.dry_run:
        downloaded_records: list[AssetRecord] = []

        if args.workers <= 1:
            for index, record in enumerate(all_records, start=1):
                log(f"[DOWNLOAD {index}/{len(all_records)}] {record.kind} {record.id} {record.image_url}")
                updated = download_asset(
                    record=record,
                    output_root=output_root,
                    limiter=limiter,
                    timeout=args.timeout,
                    retries=args.retries,
                    force=args.force,
                )
                log(f"  {updated.status} size={updated.size_bytes} path={updated.file_path} error={updated.error}")
                downloaded_records.append(updated)

                if (
                    not args.no_write_metadata
                    and args.metadata_interval > 0
                    and (index % args.metadata_interval == 0 or index == len(all_records))
                ):
                    merge_metadata(output_root, [updated])
                    log(f"[METADATA] checkpoint index={index}/{len(all_records)}")
        else:
            log(f"[DOWNLOAD MODE] parallel workers={args.workers}")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        download_asset,
                        record=record,
                        output_root=output_root,
                        limiter=limiter,
                        timeout=args.timeout,
                        retries=args.retries,
                        force=args.force,
                    ): (index, record)
                    for index, record in enumerate(all_records, start=1)
                }

                for done_count, future in enumerate(as_completed(futures), start=1):
                    index, record = futures[future]
                    try:
                        updated = future.result()
                    except Exception as exc:
                        updated = record
                        updated.status = "failed"
                        updated.error = f"{type(exc).__name__}: {exc}"
                        updated.downloaded_at = now_iso()

                    log(f"[DOWNLOAD {index}/{len(all_records)} DONE {done_count}/{len(all_records)}] {updated.kind} {updated.id}")
                    log(f"  {updated.status} size={updated.size_bytes} path={updated.file_path} error={updated.error}")
                    downloaded_records.append(updated)

                    if (
                        not args.no_write_metadata
                        and args.metadata_interval > 0
                        and (done_count % args.metadata_interval == 0 or done_count == len(all_records))
                    ):
                        merge_metadata(output_root, [updated])
                        log(f"[METADATA] checkpoint done={done_count}/{len(all_records)}")

        all_records = downloaded_records

    if not args.no_write_metadata:
        merge_metadata(output_root, all_records)
        log("[METADATA] updated metadata.csv")

    failed = [r for r in all_records if r.status == "failed"]
    missing = [r for r in all_records if r.status == "missing"]
    log("=== summary ===")
    log(f"total={len(all_records)} downloaded={sum(r.status == 'downloaded' for r in all_records)} exists={sum(r.status == 'exists' for r in all_records)} missing={len(missing)} failed={len(failed)} dry_run={args.dry_run}")

    if missing:
        log("missing sample:")
        for record in missing[:20]:
            log(f"  {record.kind} {record.id}: {record.error}")

    if failed:
        log("failed sample:")
        for record in failed[:20]:
            log(f"  {record.kind} {record.id}: {record.error}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
