"""Pure-unit tests for low-level GeoTIFF helpers.

Each module here tests a single helper or family of helpers in
isolation, without standing up a full read/write pipeline. The split
keeps the long-running integration suite separate from the fast
helper-level coverage.
"""
