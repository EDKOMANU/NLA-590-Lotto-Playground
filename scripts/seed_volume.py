"""Copy the repo's baseline data and artifacts onto a mounted volume, once.

The Fly config points LOTTO_DATA_CSV / LOTTO_ARTIFACT_DIR / LOTTO_CACHE_DIR at /data,
which starts empty. This seeds it from the copies baked into the image so the first
boot has an archive and trained models to work with. Safe to re-run: it never
overwrites a file that already exists on the volume.

    fly ssh console -C "python /app/scripts/seed_volume.py"
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lottery_core import config  # noqa: E402

BASE = config.BASE
PAIRS = [
    (os.path.join(BASE, 'ghana_lotto_history.csv'), config.CSVF),
    (os.path.join(BASE, 'artifacts'), config.ARTIFACT_DIR),
    (os.path.join(BASE, 'backtest_cache'), config.CACHE_DIR),
]


def main():
    for src, dst in PAIRS:
        if os.path.abspath(src) == os.path.abspath(dst):
            print(f"skip {dst} -- not redirected to a volume")
            continue
        if not os.path.exists(src):
            print(f"skip {dst} -- no baseline at {src}")
            continue
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            copied = 0
            for name in os.listdir(src):
                target = os.path.join(dst, name)
                if not os.path.exists(target):
                    shutil.copy2(os.path.join(src, name), target)
                    copied += 1
            print(f"{dst}: {copied} file(s) seeded")
        elif not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"{dst}: seeded")
        else:
            print(f"{dst}: already present, left alone")


if __name__ == '__main__':
    main()
