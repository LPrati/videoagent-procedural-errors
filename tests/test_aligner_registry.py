"""The aligner registry and the pure parts of its adapters.

None of the checkpoints can be loaded here — the offline environment has no
torch and no transformers, and every adapter imports them inside its methods
for exactly that reason. What can be checked without a GPU is the part that
decides *which* model runs and *what its artifact is called*, plus the
preprocessing arithmetic the adapters do in numpy: the frame average of the
image-tower aligners and ViCLIP's own tensor layout.
"""

import numpy as np
import pytest

from video_agent.adapters.aligner import (
    ALIASES,
    ALIGNERS,
    DEFAULT_ALIGNER,
    FRAMES_PER_CLIP,
    VICLIP_RESOLUTION,
    aligner_slug,
    aligner_spec,
    build_aligner,
    clip_frames,
    frame_mean,
    known_aligners,
    l2_normalize,
    pad_to_length,
    viclip_video_array,
)
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.media import Clip, Frame
from video_agent.memory.store import DEFAULT_ALIGNER_SLUG

NOT_IMPLEMENTED = (
    "facebookresearch/LaViLa-TSF-L",
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
)


def frame(index: int, value: int, size: int = 4) -> Frame:
    return Frame(
        index=index,
        time_s=index / 30.0,
        data=np.full((size, size, 3), value, dtype=np.uint8),
    )


class TestSlugs:
    def test_the_default_keeps_the_slug_the_store_names_artifacts_with(self) -> None:
        # ``memory.store`` spells this slug itself rather than importing an
        # adapter; the two have to say the same thing or the default run
        # would start writing a suffixed file.
        assert aligner_slug(DEFAULT_ALIGNER) == DEFAULT_ALIGNER_SLUG

    def test_the_project_aligner_is_a_registry_id_with_a_suffixed_artifact(
        self,
    ) -> None:
        # ``agent.config`` spells the model id without importing an adapter;
        # if the two drifted, every run would ask for a file no stage writes.
        assert DEFAULT_ALIGNER_MODEL in ALIGNERS
        assert aligner_spec(DEFAULT_ALIGNER_MODEL).kind is not None  # implemented
        assert aligner_slug(DEFAULT_ALIGNER_MODEL) == "pe-core-l"
        assert aligner_slug(DEFAULT_ALIGNER_MODEL) != DEFAULT_ALIGNER_SLUG

    def test_every_aligner_has_its_own_filesystem_safe_slug(self) -> None:
        slugs = [spec.slug for spec in ALIGNERS.values()]

        assert len(slugs) == len(set(slugs))
        for slug in slugs:
            assert slug and slug == slug.lower()
            assert set(slug) <= set("abcdefghijklmnopqrstuvwxyz0123456789-.")

    def test_the_documented_slugs(self) -> None:
        assert aligner_slug("microsoft/xclip-base-patch16") == "xclip-base-patch16"
        assert aligner_slug("timm/PE-Core-L-14-336") == "pe-core-l"
        assert aligner_slug("OpenGVLab/ViCLIP-L-14-hf") == "viclip-l-14"
        assert (
            aligner_slug("google/siglip2-so400m-patch16-384")
            == "siglip2-so400m-mean"
        )

    def test_metas_own_pe_core_ids_resolve_to_the_open_clip_weights(self) -> None:
        # facebook/PE-Core-* holds a bare .pt with no text tower, so the
        # alias has to land on the same slug and the same loadable
        # repository as the timm/ id.
        assert aligner_slug("facebook/PE-Core-L14-336") == "pe-core-l"
        assert aligner_slug("facebook/PE-Core-G14-448") == "pe-core-g"
        assert (
            aligner_spec("facebook/PE-Core-L14-336").checkpoint
            == "timm/PE-Core-L-14-336"
        )

    def test_every_alias_points_at_a_real_entry(self) -> None:
        for alias, target in ALIASES.items():
            assert target in ALIGNERS, alias

    def test_an_unknown_id_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError) as error:
            aligner_slug("microsoft/xclip-base-patch32")

        message = str(error.value)
        assert "microsoft/xclip-base-patch32" in message
        assert DEFAULT_ALIGNER in message
        assert "OpenGVLab/ViCLIP-L-14-hf" in message

    def test_known_aligners_lists_ids_and_aliases(self) -> None:
        ids = known_aligners()

        assert ids == sorted(ids)
        assert set(ids) == set(ALIGNERS) | set(ALIASES)


class TestBuildAligner:
    def test_an_unknown_id_is_refused_before_anything_is_loaded(self) -> None:
        with pytest.raises(ValueError, match="unknown aligner"):
            build_aligner("nobody/nothing", device="cpu")

    @pytest.mark.parametrize("model_name", NOT_IMPLEMENTED)
    def test_an_unimplemented_aligner_says_what_it_would_take(
        self, model_name: str
    ) -> None:
        with pytest.raises(NotImplementedError) as error:
            build_aligner(model_name, device="cpu")

        message = str(error.value)
        assert model_name in message
        # The message is the report: it has to name where the weights and
        # the modelling code come from, not just refuse.
        assert "github.com" in message

    def test_the_unimplemented_entries_still_carry_a_slug_and_a_licence(self) -> None:
        for model_name in NOT_IMPLEMENTED:
            spec = aligner_spec(model_name)
            assert spec.kind is None
            assert spec.slug and spec.licence and spec.note


class TestFrameSelection:
    def test_a_clip_is_subsampled_and_padded_to_the_frame_count(self) -> None:
        clip = Clip(frames=tuple(frame(i, i) for i in range(10)))

        chosen = clip_frames(clip, 8)

        assert len(chosen) == 8
        assert [int(f[0, 0, 0]) for f in chosen] == [0, 1, 2, 3, 5, 6, 7, 8]

    def test_a_short_clip_repeats_its_last_frame(self) -> None:
        clip = Clip(frames=(frame(0, 3), frame(1, 7)))

        chosen = clip_frames(clip, 4)

        assert [int(f[0, 0, 0]) for f in chosen] == [3, 7, 7, 7]

    def test_an_empty_clip_cannot_be_padded(self) -> None:
        with pytest.raises(ValueError, match="empty clip"):
            pad_to_length([], 8)


class TestL2Normalize:
    def test_rows_come_back_at_unit_length(self) -> None:
        rows = l2_normalize(np.array([[3.0, 4.0], [0.0, 5.0]]))

        assert rows[0] == pytest.approx([0.6, 0.8])
        assert rows[1] == pytest.approx([0.0, 1.0])

    def test_a_zero_row_stays_zero_instead_of_becoming_nan(self) -> None:
        rows = l2_normalize(np.array([[0.0, 0.0], [1.0, 0.0]]))

        assert rows[0] == pytest.approx([0.0, 0.0])
        assert not np.isnan(rows).any()

    def test_a_matrix_of_the_wrong_rank_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, D\)"):
            l2_normalize(np.zeros(4))


class TestFrameMean:
    def test_it_averages_each_clips_frames_and_normalizes(self) -> None:
        # Two clips of two frames: the first averages to (1, 1) and comes
        # back as the unit vector at 45 degrees, the second to (0, 3) -> (0, 1).
        embeddings = np.array(
            [[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 4.0]], dtype=np.float32
        )

        pooled = frame_mean(embeddings, frames_per_clip=2)

        assert pooled.shape == (2, 2)
        half = np.sqrt(0.5)
        assert pooled[0] == pytest.approx([half, half])
        assert pooled[1] == pytest.approx([0.0, 1.0])

    def test_frames_that_cancel_out_leave_a_zero_row(self) -> None:
        embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

        pooled = frame_mean(embeddings, frames_per_clip=2)

        assert pooled == pytest.approx(np.zeros((1, 2)))
        assert not np.isnan(pooled).any()

    def test_one_frame_per_clip_is_just_normalization(self) -> None:
        embeddings = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

        pooled = frame_mean(embeddings, frames_per_clip=1)

        assert pooled == pytest.approx(np.array([[0.6, 0.8], [0.0, 1.0]]))

    def test_rows_that_do_not_divide_into_clips_are_refused(self) -> None:
        with pytest.raises(ValueError, match="do not divide"):
            frame_mean(np.zeros((7, 2)), frames_per_clip=2)

    def test_a_non_positive_frame_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frames_per_clip"):
            frame_mean(np.zeros((4, 2)), frames_per_clip=0)

    def test_a_matrix_of_the_wrong_rank_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(n\*frames, D\)"):
            frame_mean(np.zeros((2, 2, 2)), frames_per_clip=2)


class TestViclipVideoArray:
    def test_frames_leave_as_normalized_bctHW(self) -> None:
        frames = [np.full((8, 6, 3), 255, dtype=np.uint8)] * 2

        array = viclip_video_array([frames])

        assert array.shape == (1, 2, 3, VICLIP_RESOLUTION, VICLIP_RESOLUTION)
        assert array.dtype == np.float32
        # A white frame is (1 - mean) / std, per channel, everywhere.
        expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        assert array[0, 0, :, 0, 0] == pytest.approx(expected, abs=1e-5)

    def test_channels_keep_their_order(self) -> None:
        # media hands over RGB; the notebook's [:, :, ::-1] only undoes
        # OpenCV's BGR read, so nothing may be flipped again here.
        frame_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_rgb[:, :, 0] = 255  # red

        array = viclip_video_array([[frame_rgb]])

        red, green, blue = array[0, 0, :, 0, 0]
        assert red > green and red > blue

    def test_clips_of_different_lengths_are_refused(self) -> None:
        frame_one = np.zeros((4, 4, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="different frame counts"):
            viclip_video_array([[frame_one], [frame_one, frame_one]])

    def test_an_empty_batch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            viclip_video_array([])


def test_every_implemented_aligner_is_held_to_the_same_frame_count() -> None:
    # The input is fixed: same grid, same 8 frames per clip, so a
    # difference between two aligners is the model and not the sampling.
    implemented = [spec for spec in ALIGNERS.values() if spec.kind is not None]

    assert implemented
    assert {spec.frames for spec in implemented} == {FRAMES_PER_CLIP}
