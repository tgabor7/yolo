from pathlib import Path

for sub in ("train", "val"):
    for lbl in sorted(Path("dataset5/labels").glob(f"{sub}/*.txt")):
        lines = lbl.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                w = float(w) * 1.2
                h = float(h) * 1.2
                new_lines.append(f"{cls} {cx} {cy} {w:.16f} {h:.16f}")
        lbl.write_text("\n".join(new_lines) + "\n")
        print(f"  {lbl}")
