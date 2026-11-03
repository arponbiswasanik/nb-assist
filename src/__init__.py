"""NB Assist — offline trilingual banking FAQ assistant.

Pipeline stages share this package. Each module owns one stage
and exposes a single entry function. Configuration lives in
config.yaml at the repository root and is passed in by the caller;
no module reads configuration directly from disk.
"""

__version__ = "0.1.0"