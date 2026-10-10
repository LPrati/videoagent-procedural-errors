"""Inference layer: the ReAct orchestrator, its tools and its transcripts.

Reproduces VideoAgent's agent loop (LangChain's ``create_react_agent`` +
``AgentExecutor``) without LangChain, so the prompt sent to the model, the
parsing of its answer and the observations fed back are ours to pin down,
version and ablate. ``runner`` holds the framework-neutral seam: swapping
``react.ReActRunner`` for another agent framework leaves ``tools`` untouched.
"""
