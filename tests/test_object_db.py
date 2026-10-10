import pytest

from video_agent.memory.object_db import (
    DEFAULT_QUERY_TEMPLATE,
    DEFAULT_SIMILARITY_THRESHOLD,
    ObjectDatabase,
    retrieve_candidate_objects,
    text_columns_for,
)

from tests.fakes import FakeSiglip, make_object_memory, object_record

N_SEGMENTS = 3

# Features concatenate a 2-column text-aligned member with a 1-column
# appearance-only one, as the reference's CLIP + DINOv2 crop embedding did.
WIDTHS = (2, 1)


@pytest.fixture
def memory():
    return make_object_memory(
        object_record(1, "person", (0, 1), [1.0, 0.0, 9.0]),
        object_record(2, "person", (2,), [0.2, 1.0, 9.0]),
        object_record(3, "cup", (0, 5), [0.5, 0.5, 9.0]),
    )


@pytest.fixture
def database(memory):
    with ObjectDatabase(memory, N_SEGMENTS) as db:
        yield db


def test_the_agents_sql_returns_rows_as_their_repr(database: ObjectDatabase) -> None:
    assert (
        database.query(
            "SELECT COUNT(DISTINCT object_id) FROM Objects WHERE category = 'person'"
        )
        == "[(2,)]"
    )
    assert (
        database.query("SELECT object_id, category FROM Objects ORDER BY object_id")
        == "[(1, 'person'), (2, 'person'), (3, 'cup')]"
    )
    assert database.query("SELECT COUNT(*) FROM Segments") == "[(3,)]"


def test_appearances_outside_the_segment_grid_are_dropped(
    database: ObjectDatabase,
) -> None:
    # Object 3 also carries segment 5, which no caption or embedding covers.
    assert (
        database.query("SELECT segment_id FROM Objects_Segments WHERE object_id = 3")
        == "[(0,)]"
    )
    assert (
        database.query(
            "SELECT object_id FROM Objects_Segments WHERE segment_id = 1 "
            "ORDER BY object_id"
        )
        == "[(1,)]"
    )


def test_broken_sql_comes_back_as_text_and_leaves_the_database_usable(
    database: ObjectDatabase,
) -> None:
    answer = database.query("SELEC * FROM Objects")

    assert isinstance(answer, str)
    assert "syntax error" in answer
    assert database.query("SELECT COUNT(*) FROM Objects") == "[(3,)]"


def test_read_only_blocks_writes_the_llm_may_emit(database: ObjectDatabase) -> None:
    answer = database.query("DELETE FROM Objects")

    assert "readonly" in answer
    assert database.query("SELECT COUNT(*) FROM Objects") == "[(3,)]"


def test_a_writable_database_is_opt_in(memory) -> None:
    with ObjectDatabase(memory, N_SEGMENTS, read_only=False) as db:
        assert db.query("DELETE FROM Objects") == "[]"
        assert db.query("SELECT COUNT(*) FROM Objects") == "[(0,)]"


def test_open_vocabulary_retrieval_keeps_what_clears_the_threshold(memory) -> None:
    embedder = FakeSiglip(keywords=("cup", "dog"))

    # Similarities to the one-hot "cup" axis: 1.0, 0.196, 0.707.
    candidates = retrieve_candidate_objects(
        memory,
        "cup",
        embedder,
        text_columns=text_columns_for(WIDTHS),
        threshold=DEFAULT_SIMILARITY_THRESHOLD,
    )

    assert candidates == [1, 3]
    assert embedder.texts == [DEFAULT_QUERY_TEMPLATE.format("cup")]
    assert embedder.texts == ["a photo of a cup."]


def test_a_lower_threshold_admits_the_weaker_match(memory) -> None:
    candidates = retrieve_candidate_objects(
        memory,
        "cup",
        FakeSiglip(),
        text_columns=text_columns_for(WIDTHS),
        threshold=0.1,
    )

    assert candidates == [1, 2, 3]


def test_open_vocabulary_retrieval_on_an_empty_memory() -> None:
    assert (
        retrieve_candidate_objects(
            make_object_memory(), "cup", FakeSiglip(), text_columns=slice(0, 2)
        )
        == []
    )


def test_text_columns_select_one_member_of_the_concatenation() -> None:
    assert text_columns_for(WIDTHS) == slice(0, 2)
    assert text_columns_for(WIDTHS, member=1) == slice(2, 3)
    with pytest.raises(ValueError):
        text_columns_for(WIDTHS, member=2)


def test_the_database_can_be_queried_and_closed_from_another_thread(memory) -> None:
    """An interactive caller may open a toolkit in one thread and use it from
    another; SQLite refuses that by default, and the loop's tools would
    fail mid-episode instead of at construction."""
    import threading

    db = ObjectDatabase(memory, N_SEGMENTS)
    outcome: dict = {}

    def use() -> None:
        try:
            outcome["rows"] = db.query("SELECT COUNT(*) FROM Objects")
            db.close()
            outcome["closed"] = True
        except Exception as error:  # surfaced by the assertion below
            outcome["error"] = error

    worker = threading.Thread(target=use)
    worker.start()
    worker.join()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["closed"] is True
