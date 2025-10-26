from pathlib import Path
import hashlib, csv
from langdetect import detect
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unk"


def main():
    files = [p for p in RAW.rglob("*") if p.is_file()]
    rows = []
    for fp in tqdm(files, desc="Scanning raw"):
        item = {
            "rel_path": str(fp.relative_to(RAW)),
            "bytes": fp.stat().st_size,
            "suffix": fp.suffix.lower(),
            "sha256": sha256(fp),
            "lang": "unk",
        }
        if fp.suffix.lower() == ".txt":
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
                item["lang"] = guess_lang(txt[:2000])
            except Exception:
                pass
        rows.append(item)
    out = OUT / "raw_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["rel_path", "bytes", "suffix", "sha256", "lang"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} files).")


if __name__ == "__main__":
    main()
