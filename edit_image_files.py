import os
import shutil
import re
from pathlib import Path

base_dir = Path("images")

max_indexs = {}

for file in base_dir.glob("*.png"):
    match = re.match(r"(\d+)-(\d+)\.png", file.name)
    if match:
        x, n = int(match[1]), int(match[2])
        max_indexs[x] = max(max_indexs.get(x, 0), n)

for file in base_dir.glob("*.PNG"):
    match = re.match(r"(\d+)-(\d+)\.PNG", file.name)
    if match:
        x, n = int(match[1]), int(match[2])
        max_indexs[x] = max(max_indexs.get(x, 0), n)

for archive_num in range(1, 5+1):
    archive_path = base_dir / f"Archive{archive_num}"
    x = archive_num
    n = max_indexs.get(x, 0) + 1

    for file in sorted(archive_path.glob("*.PNG")):
        new_name = f"{x}-{n}.png"
        new_path = base_dir / new_name
        shutil.move(str(file), str(new_path))
        print(f"Moved: {file} -> {new_path}")
        n += 1

    shutil.rmtree(base_dir / f"Archive{archive_num}")
