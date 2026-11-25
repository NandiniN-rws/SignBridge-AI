from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
VIDEO_OUT = DATASETS / "emergency_isl_videos"
IMAGE_OUT = DATASETS / "medical_isl_images"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def safe_extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        base = destination.resolve()
        for item in zf.infolist():
            target = (destination / item.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {item.filename}")
        zf.extractall(destination)

def main():
    if not DATASETS.exists():
        raise SystemExit(f"Cannot find {DATASETS}. Run this script from the SignBridge_AI project folder.")

    zip_files = list(DATASETS.rglob("*.zip"))
    if not zip_files:
        raise SystemExit("No ZIP archives found inside the datasets folder.")

    emergency_archives = [p for p in zip_files if "Cropped_Data" in [x.name for x in p.parents]]
    # Use cropped videos only; Raw_Data duplicates are intentionally not extracted.
    medical_archives = [
        p for p in zip_files
        if "Cropped_Data" not in [x.name for x in p.parents]
        and "Raw_Data" not in [x.name for x in p.parents]
        and p != Path(__file__)
    ]

    print(f"Found {len(zip_files)} ZIP files.")
    print(f"Emergency cropped archives: {len(emergency_archives)}")
    print(f"Medical image archives: {len(medical_archives)}")
    print("Raw_Data archives will be left untouched to save disk space.")

    errors = []
    for archive in emergency_archives:
        label = archive.stem.replace("_Cropped", "").replace("_cropped", "").lower()
        try:
            safe_extract(archive, VIDEO_OUT / label)
            print(f"[VIDEO] {archive.name} -> {label}")
        except Exception as e:
            errors.append((archive, str(e)))
            print(f"[ERROR] {archive.name}: {e}")

    for archive in medical_archives:
        label = archive.stem.lower()
        try:
            safe_extract(archive, IMAGE_OUT / label)
            print(f"[IMAGE] {archive.name} -> {label}")
        except Exception as e:
            errors.append((archive, str(e)))
            print(f"[ERROR] {archive.name}: {e}")

    VIDEO_OUT.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)

    video_rows = []
    for f in VIDEO_OUT.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXT:
            rel = f.relative_to(VIDEO_OUT)
            video_rows.append([str(f.resolve()), rel.parts[0], "unknown"])

    image_rows = []
    for f in IMAGE_OUT.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXT:
            rel = f.relative_to(IMAGE_OUT)
            image_rows.append([str(f.resolve()), rel.parts[0], "unknown"])

    out_dir = ROOT / "deep_learning" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if video_rows:
        with (out_dir / "manifest_emergency.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["video_path", "label", "signer_id"])
            writer.writerows(video_rows)

    if image_rows:
        with (out_dir / "manifest_medical.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["image_path", "label", "signer_id"])
            writer.writerows(image_rows)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Video files found: {len(video_rows)}")
    print(f"Video classes: {len(set(r[1] for r in video_rows))}")
    print(f"Image files found: {len(image_rows)}")
    print(f"Image classes: {len(set(r[1] for r in image_rows))}")
    print(f"Extracted videos: {VIDEO_OUT}")
    print(f"Extracted images: {IMAGE_OUT}")
    if errors:
        print(f"\nArchives with errors: {len(errors)}")
    if not video_rows:
        print("\nWARNING: No video files detected. The emergency archive contents may be image frames or nested differently.")
    if not image_rows:
        print("\nWARNING: No image files detected. Check the medical archive contents.")
    print("\nRaw ZIP archives were kept. Nothing was deleted.")

if __name__ == "__main__":
    main()
