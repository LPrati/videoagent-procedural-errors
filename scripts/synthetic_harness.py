"""Run the agent over perfect, synthetic memories to check the decision rule.

The harness tests the decision rule and the metric code against memories
built from the annotations themselves: one caption per segment, equal to the
description of the step performed then.
Seven cases perturb a normal recording — nothing, an omission, a swap of two
linked steps, a swap of two independent ones, both at once, a repeated step
and a fully shuffled timeline — and each carries the verdicts a perfect
system would emit.

If the agent fails here, the problem is the orchestrator or the answer
format, not the perception: there is nothing left to see wrong. The report
also carries η, the ratio of correct verdicts with a shuffled memory over an
ordered one: near 1 means the answers came from the recipe and not from the
video.

Needs a chat backend (``--chat openrouter`` needs the ``agent`` extra), no
GPU and no video files.

Usage:
    uv run python scripts/synthetic_harness.py \\
        --from-split data_splits/dev.json --subset normal --limit 5 \\
        --out RUNS_DIR --run-name NAME \\
        --chat vllm --model Qwen/Qwen3.5-9B --max-iterations 100 \\
        --max-tokens 0 --seed 20260903
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Mapping, Sequence

from _env import load_dotenv
from _wiring import (
    DEFAULT_MAX_TOKENS,
    LazyModels,
    react_config,
    tool_config,
)
from video_agent.adapters.aligner import aligner_slug
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.agent.runner import AgentRunner
from video_agent.eval.report import read_verdicts
from video_agent.eval.synthetic import (
    CASE_LETTERS,
    CASE_NAMES,
    DEFAULT_SEED,
    Case,
    HashingTextEmbedder,
    count_matching,
    eta,
    first_starts,
    is_order_clean,
    make_cases,
    score_case,
    synthetic_temporal_memory,
    synthetic_video,
)
from video_agent.lineage import code_version
from video_agent.task.annotations import AnnotationSet, Recording, TaskGraph, load_split
from video_agent.task.run import RunStore, TaskConfig, run_recording

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"

CAPTION_VARIANT = "synthetic"
"""Recorded as the caption source, so a harness record is never taken for a run."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--recordings", help="comma-separated recording ids to build cases from"
    )
    source.add_argument(
        "--from-split", type=Path, help="split file to take the recordings from"
    )
    parser.add_argument(
        "--subset",
        choices=["error", "normal", "all"],
        default="normal",
        help="which half of --from-split to use; the cases start from a normal "
        "recording, so 'normal' is the meaningful one",
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True, help="root of the runs")
    parser.add_argument("--run-name", default="synthetic", help="subdirectory of --out")
    parser.add_argument(
        "--cases",
        default="a,b,c,d,e,f,g",
        help="letters of the cases to run (a-g, see eval.synthetic.CASE_LETTERS)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--chat",
        choices=["openrouter", "vllm", "transformers"],
        default="vllm",
        help="orchestrator backend; 'vllm' is the locally served model "
        "(see serving/vllm/README.md)",
    )
    parser.add_argument(
        "--base-url",
        help="endpoint of the chat backend; defaults to the usual address of "
        "--chat (OpenRouter, or localhost:8000 for vllm)",
    )
    parser.add_argument("--vqa", choices=["none"], default="none")
    parser.add_argument("--vqa-style", choices=["two_questions", "single"], default="two_questions")
    parser.add_argument(
        "--aligner-model",
        default=DEFAULT_ALIGNER_MODEL,
        help="aligner the record names. The harness has no pixels, so the "
        "hashing stand-in plays both towers here; the id is recorded so a "
        "harness record carries the configuration of the run it vets",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="per-turn token cap for the orchestrator; 0 means unlimited",
    )
    parser.add_argument(
        "--chat-timeout",
        type=float,
        help="seconds one request may take; defaults to 300 for --chat vllm "
        "and to 120 elsewhere",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="let the served model reason before it answers; the reasoning "
        "block is recorded per turn and never reaches the scratchpad",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embedding-dim", type=int, default=256)
    args = parser.parse_args(argv)
    # The id only travels into the record here, but an unknown one would
    # break every case in the sweep, one identical traceback at a time.
    try:
        aligner_slug(args.aligner_model)
    except ValueError as error:
        parser.error(str(error))
    return args


def selected_cases(letters: str) -> tuple[str, ...]:
    """Translate ``"a,c,g"`` into case names, keeping the canonical order."""
    wanted = {part.strip() for part in letters.split(",") if part.strip()}
    unknown = sorted(wanted - set(CASE_LETTERS))
    if unknown:
        raise ValueError(
            f"unknown cases {unknown}, expected letters of {sorted(CASE_LETTERS)}"
        )
    chosen = {CASE_LETTERS[letter] for letter in wanted}
    return tuple(name for name in CASE_NAMES if name in chosen)


def unusable_reason(recording: Recording, graph: TaskGraph) -> str | None:
    """Why this recording cannot be the base of the cases, or ``None`` if it can.

    The cases perturb a *normal* timeline, so the base has to be one: every
    step performed, with times, in an order the task graph accepts. Some
    normal recordings perform a step before one of its prerequisites without
    the annotation tagging it; case (a) would then expect an error, which is
    not what that case is meant to test.
    """
    if any(start is None for start in first_starts(recording).values()):
        return "a step has no annotated times"
    if any(step.tags for step in recording.steps):
        return "the annotation already tags an error"
    if not is_order_clean(recording, graph):
        return "the annotated order does not respect the task graph"
    return None


def select_recordings(
    explicit: Sequence[str],
    split_ids: Sequence[str],
    annotations: AnnotationSet,
    *,
    limit: int | None = None,
) -> list[str]:
    """Choose the recordings the harness builds cases from.

    Ids given by hand are honored as written — the caller means that
    recording, and a case built on an unusual one is still informative.
    Recordings taken from a split are sorted and filtered to the ones that can
    host the cases, then truncated to ``limit``.
    """
    if explicit:
        return [recording_id.strip() for recording_id in explicit if recording_id.strip()]
    usable = []
    for recording_id in sorted(set(split_ids)):
        recording = annotations.recordings.get(recording_id)
        if recording is None:
            continue
        graph = annotations.graphs.get(recording.activity_id)
        if graph is None or unusable_reason(recording, graph) is not None:
            continue
        usable.append(recording_id)
    return usable[:limit] if limit else usable


def task_config(
    args: argparse.Namespace, prompt_template: str, question_template: str
) -> TaskConfig:
    """Freeze the harness configuration into the record, as a real run does."""
    from video_agent.agent.prompts import prompt_sha256

    return TaskConfig(
        orchestrator_model=args.model,
        chat_backend=args.chat,
        thinking=args.thinking,
        vqa_backend="none",
        vqa_style=args.vqa_style,
        caption_variant=CAPTION_VARIANT,
        use_objects=False,
        seed=args.seed,
        temperature=args.temperature,
        react=react_config(args),
        tools=tool_config(args),
        prompt_sha256=prompt_sha256(prompt_template),
        question_template_sha256=prompt_sha256(question_template),
        code_version=code_version(),
    )


def build_toolkit(case: Case, sop, embedder: HashingTextEmbedder):
    """Wire one case's synthetic memory into the agent's tools (no VQA, no objects)."""
    from video_agent.agent.tools import ToolKit

    video = synthetic_video(case.recording)
    memory = synthetic_temporal_memory(
        case.recording,
        sop,
        video=video,
        text_embedder=embedder,
        aligner=embedder,
    )
    return ToolKit(
        video=video,
        temporal=memory,
        text_embedder=embedder,
        aligner=embedder,
        objects=None,
        vqa=None,
    )


def render_report(results: Sequence[dict], case_names: Sequence[str]) -> str:
    """Render the recording × case table, plus η per recording."""
    header = "| Recording | " + " | ".join(case_names) + " | η |"
    rule = "|---" * (len(case_names) + 2) + "|"
    lines = [
        "# Synthetic harness",
        "",
        "Cells: steps whose verdict matches the expected one / total SOP steps; "
        "**ok** when all match. The `shuffled` case has no right answer: "
        "its cell counts the verdicts that stay equal to those of the `normal` case.",
        "",
        header,
        rule,
    ]
    for result in results:
        cells = []
        for name in case_names:
            score = next(
                (item for item in result["cases"] if item["case"] == name), None
            )
            if score is None:
                cells.append("—")
            elif score["matched"] is None:
                cells.append(f"{score.get('matched_vs_normal', '—')}/{score['total']}")
            else:
                mark = " ok" if score["passed"] else ""
                cells.append(f"{score['matched']}/{score['total']}{mark}")
        eta_value = result.get("eta")
        cells.append("—" if eta_value is None else f"{eta_value:.2f}")
        lines.append(f"| {result['recording_id']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Cases", ""])
    for result in results:
        for score in result["cases"]:
            note = score.get("note")
            if note:
                lines.append(f"- `{result['recording_id']}` **{score['case']}**: {note}")
    return "\n".join(lines)


def run_harness(
    *,
    recording_ids: Sequence[str],
    case_names: Sequence[str],
    annotations: AnnotationSet,
    runner: AgentRunner,
    embedder: HashingTextEmbedder,
    store: RunStore,
    prompt_template: str,
    question_template: str,
    config: TaskConfig,
    seed: int = DEFAULT_SEED,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every case of every recording and score it; return the scores.

    One agent episode per (recording, case), each over the synthetic memory
    that case's modified annotation implies, saved under
    ``<recording_id>__<case>`` so the harness records live beside real runs
    without being confused for them. A case that raises is counted as a
    missing cell rather than stopping the sweep: the harness exists to say
    *which* cases the system fails, and the first exception must not hide the
    rest.

    η rides along per recording: the shuffled case has no right answer, so it
    is scored against the verdicts the ordered memory produced. ``log``
    receives one line per case; it is a parameter so a test can
    run the sweep silently.
    """
    results: list[dict] = []
    for recording_id in recording_ids:
        recording = annotations.recordings[recording_id]
        sop, graph = annotations.for_recording(recording_id)
        cases = [
            case
            for case in make_cases(recording, sop, graph, seed=seed)
            if case.name in case_names
        ]
        normal = next((case for case in cases if case.name == "normal"), None)
        scores: list[dict] = []
        correct_normal: int | None = None
        correct_shuffled: int | None = None
        for case in cases:
            key = f"{recording_id}__{case.name}"
            started = time.monotonic()
            try:
                with build_toolkit(case, sop, embedder) as toolkit:
                    record = run_recording(
                        recording_id=key,
                        sop=sop,
                        graph=graph,
                        toolkit=toolkit,
                        runner=runner,
                        prompt_template=prompt_template,
                        question_template=question_template,
                        config=config,
                    )
                store.save(key, record)
                got = read_verdicts(record)
                score = score_case(case, got)
                score["note"] = case.note
                score["status"] = record["status"]
                if case.name == "normal" and score["matched"] is not None:
                    correct_normal = score["matched"]
                if normal is not None and normal.expected is not None:
                    matched_vs_normal, _ = count_matching(normal.expected, got)
                    score["matched_vs_normal"] = matched_vs_normal
                    if case.name == "shuffled":
                        correct_shuffled = matched_vs_normal
                scores.append(score)
                log(
                    f"[{recording_id}] {case.name}: {record['status']} "
                    f"{score['matched']}/{score['total']} "
                    f"({time.monotonic() - started:.0f}s)"
                )
            except Exception:
                log(f"[{recording_id}] {case.name}: FAILED")
                traceback.print_exc()
        results.append(
            {
                "recording_id": recording_id,
                "activity_id": recording.activity_id,
                "cases": scores,
                "eta": (
                    None
                    if correct_shuffled is None or correct_normal is None
                    else eta(correct_shuffled, correct_normal)
                ),
            }
        )
    return {
        "run_name": store.run_name,
        "seed": seed,
        "cases": list(case_names),
        "recordings": results,
    }


def write_outputs(
    store_dir: Path, scores: dict, case_names: Sequence[str]
) -> tuple[Path, Path]:
    """Write ``harness_scores.json`` and ``harness_report.md``, and name both.

    The JSON is the record — every mismatch, per step — and the Markdown is
    the summary table; they are written together so a report can
    never describe scores that were not saved.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    scores_path = store_dir / "harness_scores.json"
    report_path = store_dir / "harness_report.md"
    scores_path.write_text(
        json.dumps(scores, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        render_report(scores["recordings"], case_names) + "\n", encoding="utf-8"
    )
    return scores_path, report_path


def summarize(scores: Mapping, case_names: Sequence[str]) -> tuple[int, int, int]:
    """Count the ``(passed, scored, failed)`` cells of the table.

    A failed case left no score behind, so it is counted as the difference
    between the cells the sweep should have produced and the ones it did.
    """
    results = scores["recordings"]
    passed = sum(1 for result in results for s in result["cases"] if s.get("passed"))
    scored = sum(1 for result in results for s in result["cases"] if s["scored"])
    produced = sum(len(result["cases"]) for result in results)
    return passed, scored, len(results) * len(case_names) - produced


def _print_flushed(message: str) -> None:
    """Progress line of a sweep that takes hours; unbuffered so a tail shows it."""
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    load_dotenv()

    from video_agent.agent.prompts import load_prompt
    from video_agent.agent.react import ReActRunner

    annotations = AnnotationSet.load(args.annotations_dir)
    if args.from_split:
        ids_error, ids_normal = load_split(args.from_split)
        if args.subset == "error":
            split_ids = ids_error
        elif args.subset == "normal":
            split_ids = ids_normal
        else:
            split_ids = list(ids_error) + list(ids_normal)
        explicit: list[str] = []
    else:
        split_ids = []
        explicit = args.recordings.split(",")
    recording_ids = select_recordings(
        explicit, split_ids, annotations, limit=args.limit
    )
    case_names = selected_cases(args.cases)

    prompt_template = load_prompt("task_prompt")
    question_template = load_prompt("task_question")
    config = task_config(args, prompt_template, question_template)
    models = LazyModels(args)
    runner = ReActRunner(models.chat, react_config(args))
    store = RunStore(args.out, args.run_name)
    out = store.directory
    out.mkdir(parents=True, exist_ok=True)
    _print_flushed(
        f"{len(recording_ids)} recordings x {len(case_names)} cases -> {out}"
    )

    scores = run_harness(
        recording_ids=recording_ids,
        case_names=case_names,
        annotations=annotations,
        runner=runner,
        embedder=HashingTextEmbedder(args.embedding_dim),
        store=store,
        prompt_template=prompt_template,
        question_template=question_template,
        config=config,
        seed=args.seed,
        log=_print_flushed,
    )
    _, report_path = write_outputs(out, scores, case_names)
    passed, scored, failures = summarize(scores, case_names)
    _print_flushed(
        f"done, {passed}/{scored} scored cases passed, {failures} failures; "
        f"wrote {report_path}"
    )


if __name__ == "__main__":
    main()
