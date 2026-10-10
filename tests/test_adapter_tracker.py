from video_agent.adapters.tracker import build_category_vocabulary
from video_agent.models.types import Detection


def detection(category: str) -> Detection:
    return Detection(
        frame_index=0, box_xyxy=(0.0, 0.0, 1.0, 1.0), category=category, score=1.0
    )


def test_vocabulary_is_sorted_and_stable() -> None:
    detections = [
        [detection("cup"), detection("apple")],
        [detection("cup"), detection("bowl")],
    ]
    assert build_category_vocabulary(detections) == {"apple": 0, "bowl": 1, "cup": 2}


def test_vocabulary_empty_input() -> None:
    assert build_category_vocabulary([[], []]) == {}
