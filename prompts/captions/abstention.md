# Caption instruction: abstention

Describes only what is visible in a 2-second segment and accepts "No action
visible" as a valid caption. Same mechanics as the `action` instruction: one
2-second clip of 10 frames at 5 fps, sent as a single MP4, greedy decoding.

Read by `scripts/preprocess_temporal.py --prompt-file`. Only the `## user`
blocks reach the model; this header is documentation.

## user [middle]

This is a 2-second clip from a first-person cooking video. In one concise sentence, describe only what is visibly happening in it. Name only objects that you can actually see in the frames. Do not guess what the person is about to do, and do not assume the action that usually goes with these objects. If no action is being performed (the hands are idle, out of view, or the scene is static), answer exactly: No action visible.
