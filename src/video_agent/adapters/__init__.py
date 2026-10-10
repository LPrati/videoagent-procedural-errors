"""Concrete model adapters behind ``models.interfaces``.

Heavy dependencies (torch, transformers, ...) are imported lazily inside
each adapter so the core package works without the ``models`` extra.
"""
