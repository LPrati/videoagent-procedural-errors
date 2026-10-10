import numpy as np

from video_agent.media import Video, load_clips, load_frames, load_frames_at, open_video

from tests.conftest import FPS, FRAME_COUNT


def decoded_index(frame_data: np.ndarray) -> float:
    """Recover the index stored in the red channel (lossy codec: approximate)."""
    return float(frame_data[:, :, 0].mean())


def test_open_video(synthetic_video: Video) -> None:
    probed = open_video(synthetic_video.path)
    assert round(probed.fps) == FPS
    assert probed.frame_count == FRAME_COUNT


def test_load_frames_samples_at_requested_fps(synthetic_video: Video) -> None:
    frames = load_frames(synthetic_video, sample_fps=15)
    assert [f.index for f in frames] == list(range(0, FRAME_COUNT, 2))
    assert frames[1].time_s == 2 / FPS
    for frame in frames[:5]:
        assert abs(decoded_index(frame.data) - frame.index) < 3


def test_load_frames_interval_never_below_native(synthetic_video: Video) -> None:
    frames = load_frames(synthetic_video, sample_fps=1000)
    assert [f.index for f in frames] == list(range(FRAME_COUNT))


def test_load_clips_contiguous_intervals(synthetic_video: Video) -> None:
    clips = load_clips(synthetic_video, [(0.0, 2.0), (2.0, 4.0)])
    assert [f.index for f in clips[0].frames] == list(range(0, 60))
    assert [f.index for f in clips[1].frames] == list(range(60, 120))
    assert abs(decoded_index(clips[1].frames[0].data) - 60) < 3


def test_load_clips_subsampled(synthetic_video: Video) -> None:
    clips = load_clips(synthetic_video, [(0.0, 2.0)], sample_fps=5)
    assert [f.index for f in clips[0].frames] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_load_frames_windows_concatenate_to_whole_video(synthetic_video: Video) -> None:
    whole = [f.index for f in load_frames(synthetic_video, sample_fps=10)]
    windows = [
        f.index
        for start in range(0, FRAME_COUNT, 50)
        for f in load_frames(synthetic_video, sample_fps=10, frame_range=(start, start + 50))
    ]
    assert windows == whole  # grid anchored at frame 0, not at the window


def test_load_frames_at_decodes_exact_indices(synthetic_video: Video) -> None:
    frames = load_frames_at(synthetic_video, [90, 7, 7, 33])
    assert [f.index for f in frames] == [7, 33, 90]
    for frame in frames:
        assert abs(decoded_index(frame.data) - frame.index) < 3


def test_load_clips_seeks_backwards_and_clamps_to_end(synthetic_video: Video) -> None:
    clips = load_clips(synthetic_video, [(2.0, 4.0), (0.0, 1.0), (4.0, 99.0)])
    assert [f.index for f in clips[1].frames] == list(range(0, 30))
    assert [f.index for f in clips[2].frames] == list(range(120, FRAME_COUNT))
