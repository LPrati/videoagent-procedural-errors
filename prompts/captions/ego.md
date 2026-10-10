# Caption instruction: EgoSchema (`#C` prefix)

A variant of the `action` instruction for EgoSchema. The benchmark's
questions call the camera wearer "C", and the original multiple-choice
orchestrator prompt explains that captions starting with `#C` describe the
camera wearer and captions starting with `#O` describe someone else (the
caption format of LaViLa). This instruction asks for that format. 2-second
clip without context (`--context-segments 0`), greedy decoding.

Read by `scripts/preprocess_temporal.py --prompt-file`. Only the `## user`
block reaches the model; this header is documentation.

## user

This is a 2-second clip from a first-person video. The person wearing the camera is called C. In one concise sentence, describe what is happening in the clip and mention the objects being manipulated. Start the sentence with "#C C" when it describes what C does (for example: "#C C picks up the cup from the table."), and with "#O" followed by who acts when it describes what another person does (for example: "#O A man opens the door."). If nobody is doing anything, describe what C is looking at, starting with "#C C looks at".
