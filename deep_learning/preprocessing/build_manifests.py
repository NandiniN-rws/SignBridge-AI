
"""Build manifests from class-folder datasets.
Expected layout: ROOT/<class_name>/**/*.(mp4|avi|mov|jpg|png|jpeg)
If the downloaded archive has a different layout, inspect it and create the CSV manually
using deep_learning/data/manifest_template.csv.
"""
import argparse,csv
from pathlib import Path
VIDEO={".mp4",".avi",".mov",".mkv",".webm"}
IMAGE={".jpg",".jpeg",".png",".bmp",".webp"}
def build(root,out,extensions):
    root=Path(root);out=Path(out)
    if not root.exists(): raise SystemExit(f"Folder not found: {root}")
    rows=[]
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            rel=f.relative_to(root)
            if len(rel.parts)<2:
                print("Skipping file not inside a class folder:",f);continue
            label=rel.parts[0]
            # Keep participant folder as signer where the structure is class/signer/file.
            signer=rel.parts[1] if len(rel.parts)>=3 else "unknown"
            rows.append((str(f.resolve()),label,signer))
    if not rows: raise SystemExit(f"No supported files found under {root}. Check extraction/layout.")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as fp:
        w=csv.writer(fp);w.writerow(["video_path" if extensions==VIDEO else "image_path","label","signer_id"]);w.writerows(rows)
    print(f"Saved {len(rows)} items, {len(set(r[1] for r in rows))} classes -> {out}")
    print("Classes:",", ".join(sorted(set(r[1] for r in rows))))
def main():
    p=argparse.ArgumentParser();p.add_argument("--videos-root");p.add_argument("--images-root")
    a=p.parse_args()
    if a.videos_root:build(a.videos_root,"deep_learning/data/manifest_emergency.csv",VIDEO)
    if a.images_root:build(a.images_root,"deep_learning/data/manifest_medical.csv",IMAGE)
    if not a.videos_root and not a.images_root:p.error("Provide --videos-root and/or --images-root")
if __name__=="__main__":main()
