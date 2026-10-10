"""Turning a recipe and a video into per-step procedural verdicts.

The inference layer (``video_agent.agent``) answers one free-text question
about one video; the evaluation asks a structured one: for every step of the
recipe's SOP, was it executed, where, and was its order respected. This
package holds the three pieces that connect the two: the annotations that
define the SOP and the task graph, the question and answer format the model
sees, and the run record.

Nothing here passes the labels a system is scored against to the model:
``annotations`` loads error tags and annotated times because the evaluation
needs them, and because normal recordings supply the canonical step order,
but a run only ever receives the SOP, the task graph and the video.
"""
