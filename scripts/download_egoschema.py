"""Download the 500 clips of the public EgoSchema subset.

The EgoSchema repository (https://github.com/egoschema/EgoSchema) provides
``questions.json``, with a ``google_drive_id`` per question, and
``subset_answers.json``, the answers of the 500 public questions. This script
downloads the clip of every question in ``subset_answers.json`` from Google
Drive into ``--out``, named ``<q_uid>.mp4``. It resumes: a clip already on
disk is skipped. Failed downloads are listed at the end; run it again to
retry them.

Needs ``gdown``, which is not a dependency of the project:

    uv run --with gdown python scripts/download_egoschema.py \\
        --questions EGOSCHEMA_REPO/questions.json \\
        --answers EGOSCHEMA_REPO/subset_answers.json --out EGOSCHEMA_VIDEOS

The clips are used as downloaded (3 minutes each); the memory is built with
the same scripts as for CaptainCook4D.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

MIN_BYTES = 1_000_000
"""A smaller file is an interrupted download or an error page, not a clip."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import gdown

    questions = {
        item["q_uid"]: item
        for item in json.loads(args.questions.read_text(encoding="utf-8"))
    }
    uids = sorted(json.loads(args.answers.read_text(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for index, uid in enumerate(uids, 1):
        target = args.out / f"{uid}.mp4"
        if target.exists() and target.stat().st_size > MIN_BYTES:
            continue
        url = f"https://drive.google.com/uc?id={questions[uid]['google_drive_id']}"
        for attempt in range(args.attempts):
            try:
                gdown.download(url, str(target), quiet=True)
            except Exception as error:  # network errors, quota pages
                print(f"{uid}: attempt {attempt + 1} failed: {error}", flush=True)
            if target.exists() and target.stat().st_size > MIN_BYTES:
                break
            time.sleep(5 * (attempt + 1))
        if target.exists() and target.stat().st_size > MIN_BYTES:
            print(f"[{index}/{len(uids)}] {uid}", flush=True)
        else:
            target.unlink(missing_ok=True)
            failed.append(uid)
            print(f"[{index}/{len(uids)}] {uid} FAILED", flush=True)
    print(f"{len(uids) - len(failed)} of {len(uids)} clips on disk", flush=True)
    if failed:
        print("failed: " + " ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
