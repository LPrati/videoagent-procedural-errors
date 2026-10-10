"""Answer the EgoSchema subset with the agent: one run record per question.

Every question of ``subset_answers.json`` is asked over its own preprocessed clip with the packaged
``multiple_choice_prompt`` — the original's wording, unedited — and the
answer is read as the reference read it: the label after ``Final Answer:``.
Nothing is scored here; ``evaluate_egoschema.py`` reads the records.

The batch resumes and shards like ``run_task.py``: a question that already
has a record is skipped unless ``--force``, ``--shard I/N`` splits the work
deterministically, and a question that blows up is written down as an error
rather than taking the other 499 with it.

Requires the ``agent`` extra for the API chat backend and the ``models``
extra for anything local (uv sync --extra agent --extra models).

Usage:
    uv run python scripts/run_egoschema.py \\
        --questions EGOSCHEMA_DIR/questions.json \\
        --answers EGOSCHEMA_DIR/subset_answers.json \\
        --videos-dir VIDEOS_DIR --artifacts-root PREPROCESS_DIR \\
        --out RUNS_DIR --run-name NAME \\
        --chat vllm --model Qwen/Qwen3.5-9B --vqa api --device cpu \\
        --caption-variant ego --max-iterations 15
"""

import argparse
import hashlib
import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from _env import load_dotenv
from _wiring import LazyModels, build_toolkit, check_object_memory_slug, react_config
from ask import add_model_arguments, run_config
from video_agent.adapters.aligner import aligner_slug
from video_agent.lineage import code_version
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.task.run import CONFIG_ID_LENGTH, RUN_FILE, RunStore

PROMPT_NAME = "multiple_choice_prompt"
"""The original's multiple-choice prompt, used unedited."""

QUESTION_FORMAT_VERSION = 1
"""Bumped if the string the model is asked ever stops being ``build_question``."""

N_OPTIONS = 5
"""EgoSchema labels its choices 0 to 4; a question with fewer is malformed."""

OPTION_KEYS = tuple(f"option {index}" for index in range(N_OPTIONS))
"""Keys of the five choices in ``questions.json``, as the benchmark ships them."""

PREDICTIONS_FILE = "predictions.json"
"""``{q_uid: label}``, the shape the EgoSchema validation server accepts."""

INVALID_FILE = "invalid.json"
"""The questions left out of ``predictions.json``, with why."""

VALID_STATUSES = ("ok", "max_iterations", "parse_failed", "error", "missing")
"""Every ``status`` a record can carry; ``ok`` is the only one with a label.

The four of ``run_task.py`` plus ``missing``, which is not a failure of the
agent but of the preprocessing: the clip has no memory directory, so no
episode was ever started. It counts as wrong all the same.
"""

_FINAL_ANSWER = "Final Answer:"

_CHOICE = re.compile(
    r"^\W*(?:(?:the|correct|best|is|final|answer|choice|option|label)\W+)*([0-4])(?!\d)",
    re.IGNORECASE,
)
"""What counts as a label: a single digit 0-4 where the answer starts.

The reference asked for ``Final Answer: 3`` and took whatever followed, so
the leading noise a model adds — brackets, "choice", "option", a stray
period — is read as the same answer. Anything else is invalid and is
recorded as such: a guess here would be a number the agent never gave
(invalid answers count as wrong).
"""

_JSON_START = ("{", "[")
"""An answer opening a JSON value is never a label, whatever digit is in it.

A model can answer with a JSON object instead of a label, and the filler
words above would happily read ``{"answer": 2}`` as the label 2 — a number
the agent wrote as a field value and not as a choice.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--questions", type=Path, required=True, help="questions.json of EgoSchema"
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="subset_answers.json; its keys are the set that is run",
    )
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="root of the runs")
    parser.add_argument("--run-name", required=True, help="subdirectory of --out")
    parser.add_argument(
        "--question-list",
        type=Path,
        help="JSON list of q_uid to run instead of the whole subset (e.g. a "
        "sample, or the questions of a paired comparison); every id must be "
        "in --answers",
    )
    parser.add_argument("--limit", type=int, help="stop after this many questions")
    parser.add_argument(
        "--shard", help="I/N: process only questions where index %% N == I"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun questions that already have a record",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="rerun questions whose record has status 'error'",
    )
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    # A typo in a model id or a memory slug is not a run: refuse it here,
    # hours before 500 questions fail one by one on the same name.
    try:
        aligner_slug(args.aligner_model)
        check_object_memory_slug(args.object_memory)
    except ValueError as error:
        parser.error(str(error))
    return args


def parse_choice(answer_text: str | None) -> int | None:
    """The label the agent answered with, or ``None`` when there is none.

    Takes the *last* ``Final Answer:`` of the text, because a completion can
    quote the format before using it; text that never says it is read as the
    answer itself, which is what ``AgentResult.answer`` already holds (the
    ReAct parser splits on the same marker).
    """
    if not answer_text:
        return None
    tail = answer_text.rsplit(_FINAL_ANSWER, 1)[-1].lstrip()
    if tail.startswith(_JSON_START):
        return None
    match = _CHOICE.match(tail)
    return None if match is None else int(match.group(1))


def load_questions(path: Path) -> dict[str, dict]:
    """``questions.json`` keyed by ``q_uid``, with its five options checked."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not hold a JSON list of questions")
    questions: dict[str, dict] = {}
    for entry in data:
        q_uid = str(entry["q_uid"])
        missing = [key for key in OPTION_KEYS if key not in entry]
        if missing:
            raise ValueError(f"{path}: {q_uid} has no {', '.join(missing)}")
        questions[q_uid] = entry
    return questions


def load_answers(path: Path) -> dict[str, int]:
    """``subset_answers.json`` as ``q_uid -> label``, the set that is run."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object of answers")
    answers = {str(key): int(value) for key, value in data.items()}
    out_of_range = sorted(
        key for key, value in answers.items() if not 0 <= value < N_OPTIONS
    )
    if out_of_range:
        raise ValueError(f"{path}: labels outside 0-4 for {', '.join(out_of_range)}")
    return answers


def load_question_list(path: Path) -> list[str]:
    """A JSON list of ``q_uid``: a sample, or a paired comparison's questions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not hold a JSON list of q_uid")
    return [str(item) for item in data]


def options_of(entry: dict) -> list[str]:
    """The five choices of one question, in label order."""
    return [str(entry[key]) for key in OPTION_KEYS]


def build_question(question: str, options: Sequence[str]) -> str:
    """The ``{input}`` of the prompt: the question, then the labelled choices.

    Fixed format (``question_format_version``): the model is asked for one of
    the labels it is shown, so the labels have to be in the text and nowhere
    else. A change here is a change of configuration and needs the version.
    """
    lines = [str(question).strip()]
    lines.extend(
        f"{label}: {str(option).strip()}" for label, option in enumerate(options)
    )
    return "\n".join(lines)


def select_questions(
    answers: dict[str, int], args: argparse.Namespace, store: RunStore
) -> list[str]:
    """Which questions this process runs, in which order.

    Sorted first so that sharding is a property of the subset and not of the
    file's key order: worker ``I`` of ``N`` always gets the same questions.
    Resumption is decided after sharding and before ``--limit``, so a second
    pass with the same flags picks up exactly what is missing.
    """
    q_uids = set(answers)
    if args.question_list is not None:
        wanted = load_question_list(args.question_list)
        unknown = sorted(set(wanted) - q_uids)
        if unknown:
            raise ValueError(
                f"{args.question_list} names {len(unknown)} question(s) that are "
                f"not in --answers: {', '.join(unknown[:5])}"
            )
        q_uids = set(wanted)
    selected = sorted(q_uids)

    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        if total <= 0 or not 0 <= index < total:
            raise ValueError(f"--shard {args.shard} is not a valid I/N with 0 <= I < N")
        selected = selected[index::total]

    if not args.force:
        selected = [
            q_uid
            for q_uid in selected
            if not store.has(q_uid)
            or (args.retry_errors and store.load(q_uid).get("status") == "error")
        ]

    if args.limit:
        selected = selected[: args.limit]
    return selected


def egoschema_config(args: argparse.Namespace, prompt_template: str) -> dict:
    """This run's configuration: ``ask.py``'s, plus what the batch fixes.

    ``prompt_name`` and ``question_format_version`` are in here and not only
    in the record because both decide an answer — which wording the model
    read, and how the five choices were laid out in front of it — so both
    have to move ``config_id``.
    """
    return {
        **run_config(args, prompt_template),
        "prompt_name": PROMPT_NAME,
        "question_format_version": QUESTION_FORMAT_VERSION,
    }


def config_id(config: dict) -> str:
    """Short digest naming a configuration, as ``task.run.config_id`` does.

    Keys are sorted so the id depends on the values and not on the order the
    dict happens to serialize them in.
    """
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]


def _now() -> str:
    """UTC timestamp at second resolution, as the record stores it."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_record(
    *,
    q_uid: str,
    question: str,
    options: Sequence[str],
    truth: int,
    config: dict,
    started_at: str,
    seconds: float,
    result=None,
    error: str | None = None,
    partial=None,
    status: str | None = None,
) -> dict:
    """Turn one episode — finished, failed or never started — into a record.

    Nothing here touches a model or a tool, so the record a batch writes can
    be rebuilt, and asserted on, from a transcript alone. ``status`` is only
    passed for an episode that never ran (``missing``); otherwise it follows
    from what came back: a label read is ``ok``, an agent that finished
    without one is ``parse_failed``, and everything else keeps the loop's own
    verdict (``max_iterations``, ``error``).
    """
    from video_agent.models.types import Usage
    from video_agent.task.run import total_usage

    answer_raw = None if result is None else result.answer
    prediction = parse_choice(answer_raw)
    if status is None:
        if result is None:
            status = "error"
        elif prediction is not None:
            status = "ok"
        elif result.status == "final_answer":
            status = "parse_failed"
        else:
            status = result.status

    transcript = result.transcript if result is not None else partial
    usage = Usage() if transcript is None else transcript.usage
    usage_sum = Usage() if transcript is None else total_usage(transcript)
    return {
        "q_uid": q_uid,
        "question": question,
        "options": list(options),
        "prediction": prediction,
        "answer_raw": answer_raw,
        "truth": truth,
        # ``None`` and not ``False``: an invalid answer is scored wrong by the
        # evaluator, but the record must not claim the agent answered
        # something and missed.
        "correct": None if prediction is None else prediction == truth,
        "status": status,
        "error": error,
        "config": {**config, "config_id": config_id(config)},
        "started_at": started_at,
        "finished_at": _now(),
        "seconds": seconds,
        "usage": usage.to_dict(),
        "usage_total": usage_sum.to_dict(),
        "transcript": None if transcript is None else transcript.to_dict(),
        "code_version": config.get("code_version", code_version()),
    }


def run_question(
    *,
    q_uid: str,
    entry: dict,
    truth: int,
    args: argparse.Namespace,
    models: LazyModels,
    runner,
    prompt_template: str,
    config: dict,
) -> dict:
    """Ask one question over one clip and return its record.

    A failure of one question is recorded, not raised: 500 questions must not
    stop because one clip's tools blew up, and a record with status
    ``error`` is itself a result — one that keeps the steps the agent had
    already run, when the runner hands its partial transcript over.
    """
    from video_agent.agent.transcript import Transcript
    from video_agent.memory.store import load_video_meta

    options = options_of(entry)
    question = build_question(entry.get("question", ""), options)
    started_at = _now()
    started = time.perf_counter()

    def record(**kwargs) -> dict:
        return make_record(
            q_uid=q_uid,
            question=question,
            options=options,
            truth=truth,
            config=config,
            started_at=started_at,
            seconds=time.perf_counter() - started,
            **kwargs,
        )

    # Read directly and not through ``VideoArtifacts``, whose constructor
    # creates the directory it is asked about — which would turn "this clip
    # was never preprocessed" into "this clip has an empty memory".
    if not (Path(args.artifacts_root) / q_uid).is_dir():
        return record(status="missing", error=f"no memory directory for {q_uid}")

    toolkit = None
    try:
        artifacts = VideoArtifacts(args.artifacts_root, q_uid)
        video = load_video_meta(artifacts, Path(args.videos_dir) / f"{q_uid}.mp4")
        toolkit = build_toolkit(video, artifacts, models, args, runner)
        result = runner.run(
            question, toolkit.main_tools(), prompt_template=prompt_template
        )
        return record(result=result)
    except Exception as exception:
        carried = getattr(exception, "transcript", None)
        return record(
            error=f"{type(exception).__name__}: {exception}",
            partial=carried if isinstance(carried, Transcript) else None,
        )
    finally:
        if toolkit is not None:
            toolkit.close()


def collect_predictions(run_dir: Path) -> dict[str, int | None]:
    """Every record's label, read back from disk, ``None`` where invalid.

    Rebuilt from the run directory rather than accumulated in memory so that
    two shards, or a resumed run, produce the same aggregate as one process
    would have: whatever is on disk is the answer.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return {}
    predictions: dict[str, int | None] = {}
    for path in sorted(run_dir.glob(f"*/{RUN_FILE}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        q_uid = str(record.get("q_uid") or path.parent.name)
        prediction = record.get("prediction")
        predictions[q_uid] = None if prediction is None else int(prediction)
    return predictions


def invalid_rows(run_dir: Path) -> list[dict]:
    """One row per question with no label: what it was and why.

    The counterpart of ``predictions.json``: the validation server takes only
    the questions that were answered, and the ones that were not are a result
    too (the invalid rate is reported by cause).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    rows = []
    for path in sorted(run_dir.glob(f"*/{RUN_FILE}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("prediction") is not None:
            continue
        rows.append(
            {
                "q_uid": str(record.get("q_uid") or path.parent.name),
                "status": record.get("status"),
                "answer_raw": record.get("answer_raw"),
                "error": record.get("error"),
            }
        )
    return rows


def write_predictions(run_dir: Path) -> tuple[Path, Path]:
    """Write ``predictions.json`` and ``invalid.json`` from the records."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = collect_predictions(run_dir)
    answered = {
        q_uid: label for q_uid, label in predictions.items() if label is not None
    }
    predictions_path = run_dir / PREDICTIONS_FILE
    invalid_path = run_dir / INVALID_FILE
    predictions_path.write_text(
        json.dumps(answered, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    invalid_path.write_text(
        json.dumps(invalid_rows(run_dir), indent=1) + "\n", encoding="utf-8"
    )
    return predictions_path, invalid_path


def main() -> None:
    args = parse_args()
    load_dotenv()

    from video_agent.agent.prompts import load_prompt
    from video_agent.agent.react import ReActRunner

    questions = load_questions(args.questions)
    answers = load_answers(args.answers)
    store = RunStore(args.out, args.run_name)
    selected = select_questions(answers, args, store)

    unknown = [q_uid for q_uid in selected if q_uid not in questions]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} answered question(s) are not in {args.questions}: "
            f"{', '.join(unknown[:5])}"
        )

    prompt_template = load_prompt(PROMPT_NAME)
    config = egoschema_config(args, prompt_template)
    models = LazyModels(args)
    runner = ReActRunner(models.chat, react_config(args))
    print(
        f"{len(selected)} questions -> {store.directory} (model={args.model}, "
        f"config {config_id(config)}, code {config.get('code_version')})",
        flush=True,
    )

    for position, q_uid in enumerate(selected, 1):
        try:
            record = run_question(
                q_uid=q_uid,
                entry=questions[q_uid],
                truth=answers[q_uid],
                args=args,
                models=models,
                runner=runner,
                prompt_template=prompt_template,
                config=config,
            )
            store.save(q_uid, record)
            usage = record["usage_total"]
            print(
                f"[{position}/{len(selected)}] {q_uid}: {record['status']} "
                f"pred={record['prediction']} truth={record['truth']} "
                f"({record['seconds']:.0f}s) llm={usage['llm_calls']} "
                f"tools={usage['tool_calls']} "
                f"tokens={usage['tokens_in']}+{usage['tokens_out']}",
                flush=True,
            )
        except Exception:
            # Only a failure of the record itself reaches here: the episode's
            # own failures are records with status 'error'.
            print(f"[{position}/{len(selected)}] {q_uid}: FAILED", flush=True)
            traceback.print_exc()

    predictions_path, invalid_path = write_predictions(store.directory)
    print(f"wrote {predictions_path}", flush=True)
    print(f"wrote {invalid_path}", flush=True)


if __name__ == "__main__":
    main()
