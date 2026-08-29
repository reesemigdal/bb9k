#!/usr/bin/env python3
"""Fire-event logging: a size-capped JSONL event log plus a size-capped image dump.

EventLogger appends one JSON record per fire event (detection/aim/result
data) to a JSONL file partitioned by date, deleting its own oldest whole
files if the total grows past a configured size cap. ImageLogger saves the
frame from that same event to a directory, pruning its own oldest files if
the directory grows past a configured size cap.
"""

import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

DEFAULT_MAX_BYTES = 1_073_741_824  # 1 GiB
DEFAULT_PRUNE_FRACTION = 0.25


class EventLogger:
    """Appends one JSON record per fire event to a date-partitioned JSONL file.

    path's stem/suffix (e.g. 'events.jsonl') become a naming pattern - each
    calendar day's events go to their own file, f'{stem}_{YYYY_MM_DD}
    {suffix}' (e.g. 'events_2026_08_29.jsonl') in path's directory, rolling
    over to a new file at midnight. Once the total size of these files
    exceeds max_bytes, the oldest whole files are deleted (never rewritten
    or truncated) until prune_fraction of max_bytes has been freed.
    """

    def __init__(self, path, max_bytes=DEFAULT_MAX_BYTES, prune_fraction=DEFAULT_PRUNE_FRACTION):
        path = Path(path)
        self.directory = path.parent
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stem = path.stem
        self.suffix = path.suffix
        self.max_bytes = max_bytes
        self.prune_fraction = prune_fraction

    def _file_for(self, dt):
        return self.directory / f"{self.stem}_{dt.strftime('%Y_%m_%d')}{self.suffix}"

    def _our_files(self):
        return sorted(self.directory.glob(f'{self.stem}_*{self.suffix}'))

    def log(self, **fields):
        """Write one record (current timestamp + fields) as a JSON line.

        fields is whatever the caller wants recorded (detection data, aim
        solution, fire outcome, ...) - just make sure every value is
        JSON-serializable (floats/ints/str/bool/None/list/dict, not numpy
        scalars or arrays). Appended to today's file (see _file_for),
        creating it if today is the first event since the last rollover.
        """
        now = datetime.now()
        record = {'timestamp': now.isoformat(), **fields}
        with open(self._file_for(now), 'a') as f:
            f.write(json.dumps(record) + '\n')
        self._prune()
        return record

    def _prune(self):
        """Delete oldest whole day-files until usage is back under max_bytes.

        Never rewrites or truncates a file - only ever deletes one
        entirely - and frees prune_fraction of max_bytes in one pass (not
        just enough to dip back under the cap) so a steady stream of
        events doesn't trigger a delete on every single log() call.
        """
        files = self._our_files()
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        if total <= self.max_bytes:
            return

        target = self.max_bytes * (1 - self.prune_fraction)
        for f, size in zip(files, sizes):
            if total <= target:
                break
            f.unlink()
            total -= size


class ImageLogger:
    """Saves fire-event frames to a directory, capped at a total size.

    Files are named f'{prefix}_{timestamp}_{seq}.{ext}' - the prefix marks
    them as this logger's (pruning only ever touches files matching prefix
    and ext, so other files in the directory are left alone), the
    microsecond timestamp + monotonic seq keep names unique (never
    clobbering a previous save), and lexicographic order on the name is
    also chronological order, so the oldest files are always the ones
    sorted first.
    """

    def __init__(self, directory, max_bytes=DEFAULT_MAX_BYTES,
                 prune_fraction=DEFAULT_PRUNE_FRACTION, prefix='bb9k', ext='jpg'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.prune_fraction = prune_fraction
        self.prefix = prefix
        self.ext = ext
        self._seq = itertools.count()

    def _our_files(self):
        return sorted(self.directory.glob(f'{self.prefix}_*.{self.ext}'))

    def save(self, image, timestamp=None):
        """Write image (a cv2-compatible ndarray) under a fresh filename.

        Prunes this logger's oldest files first if that would leave the
        directory over max_bytes. Returns the filename written (str, not
        a full path).
        """
        if timestamp is None:
            timestamp = time.time()
        stamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%dT%H%M%S_%f')
        name = f'{self.prefix}_{stamp}_{next(self._seq):04d}.{self.ext}'
        cv2.imwrite(str(self.directory / name), image)
        self._prune()
        return name

    def _prune(self):
        """Delete oldest files until usage is back under max_bytes.

        Only kicks in once max_bytes is exceeded, and then frees
        prune_fraction of max_bytes in one pass (not just enough to dip
        back under the cap) so a steady stream of saves doesn't trigger a
        delete on every single one.
        """
        files = self._our_files()
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        if total <= self.max_bytes:
            return

        target = self.max_bytes * (1 - self.prune_fraction)
        for f, size in zip(files, sizes):
            if total <= target:
                break
            f.unlink()
            total -= size
