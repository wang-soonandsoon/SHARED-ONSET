"""Read-only source MIDI adapters and reproducible per-work dataset preparation."""

from .midi import GridTrack, GridWindow, MidiGridError, WindowCandidates, extract_windows, read_melody, read_window_candidates, write_grid_midi

__all__ = ["GridTrack", "GridWindow", "MidiGridError", "WindowCandidates", "extract_windows", "read_melody", "read_window_candidates", "write_grid_midi"]
