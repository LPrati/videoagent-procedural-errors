# Caption instruction: abstention with context

The abstention instruction applied to a 6-second clip: the previous segment,
the target segment and the next one (30 frames at 5 fps, one MP4), with
`--context-segments 1`. The caption describes only the 2 seconds in the
middle. At the first and last segment of a recording the clip is 4 seconds
long, and the `first` and `last` blocks are used instead; a recording with a
single segment uses `first`.

Read by `scripts/preprocess_temporal.py --prompt-file`, one `## user [tag]`
block per clip position (`middle`, `first`, `last`). Only the blocks reach
the model; this header is documentation.

## user [middle]

This is a 6-second clip from a first-person cooking video. The first two seconds and the last two seconds are context only. In one concise sentence, describe only what is visibly happening between second 2 and second 4 of the clip. Name only objects that you can actually see in the frames. Do not guess what the person is about to do, and do not assume the action that usually goes with these objects. If no action is being performed in those two seconds (the hands are idle, out of view, or the scene is static), answer exactly: No action visible.

## user [first]

This is a 4-second clip from a first-person cooking video. The last two seconds are context only. In one concise sentence, describe only what is visibly happening in the first two seconds of the clip. Name only objects that you can actually see in the frames. Do not guess what the person is about to do, and do not assume the action that usually goes with these objects. If no action is being performed in those two seconds (the hands are idle, out of view, or the scene is static), answer exactly: No action visible.

## user [last]

This is a 4-second clip from a first-person cooking video. The first two seconds are context only. In one concise sentence, describe only what is visibly happening in the last two seconds of the clip. Name only objects that you can actually see in the frames. Do not guess what the person is about to do, and do not assume the action that usually goes with these objects. If no action is being performed in those two seconds (the hands are idle, out of view, or the scene is static), answer exactly: No action visible.
