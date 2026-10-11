"""Build a manifest only when image/mask stems and patient regex are verified."""

import argparse
import csv
import re
from pathlib import Path


def nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def collect(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or not path.name.endswith((".nii", ".nii.gz")):
            continue
        stem = nifti_stem(path)
        if stem in result:
            raise ValueError(f"Repeated Nifti stem {stem}: {result[stem]}, {path}")
        result[stem] = path.resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--patient-regex", required=True,
                        help="Regex with one capture group that extracts patient ID from case stem")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    images, masks = collect(args.image_dir), collect(args.mask_dir)
    if not images or images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())[:10]
        missing_images = sorted(masks.keys() - images.keys())[:10]
        raise ValueError(f"Unpaired cases; missing masks {missing_masks}, missing images {missing_images}")
    regex = re.compile(args.patient_regex)
    rows = []
    for stem in sorted(images):
        match = regex.search(stem)
        if match is None or len(match.groups()) != 1:
            raise ValueError(f"Patient regex must capture exactly one ID from {stem}")
        rows.append({"case_id": stem, "patient_id": match.group(1),
                     "image": str(images[stem]), "mask": str(masks[stem])})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "patient_id", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases from {len({r['patient_id'] for r in rows})} patients to {args.out}")


if __name__ == "__main__":
    main()
