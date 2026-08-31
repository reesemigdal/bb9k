#!/usr/bin/env python3
"""Motion detection via frame differencing on a small, subsampled frame.

MotionDetector keeps state (the previous downsampled frame) across calls to
process(), which takes a full-size camera frame, subsamples it down to a
small working resolution (default 320x240) into a statically allocated
buffer, and diffs it against the frame from the previous call. Connected-
component analysis on the thresholded diff, filtered by a minimum area,
gives get_motion_regions() its bounding boxes. The threshold itself floats
above a static floor (diff_thresh) to track a noise estimate that adapts
over ~10s (noise_window_s), so sensor noise on a noisy night doesn't read
as motion - see the "dynamic threshold" note on MotionDetector.
"""

import math
import time

import cv2
import numpy as np

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_DIFF_THRESH = 25
DEFAULT_MIN_AREA = 500
DEFAULT_GAP_SPAN = 5
DEFAULT_NOISE_WINDOW_S = 10.0
DEFAULT_NOISE_MULTIPLIER = 2.0


class MotionDetector:
    """Frame-differencing motion detector, run on a small, fixed-size frame.

    process() resizes each incoming full-size frame (nearest-neighbor, via
    cv2.resize) down to (width, height) - default 320x240 - writing
    directly into a buffer allocated once at construction, so repeated
    calls do no per-frame allocation for the resize itself. The only state
    kept across calls is prev_frame, a copy of that same small frame from
    the previous process() call, which the current one is diffed against.
    The first call has no prev_frame yet to diff against, so it reports no
    motion (regions stay empty) - it just seeds state for the next call.

    Before connected-component analysis, the thresholded mask is closed
    (dilate then erode, cv2.MORPH_CLOSE) with a gap_span-sized kernel, so
    two blobs of motion separated by a gap of up to ~gap_span px merge into
    one component instead of being counted (or area-filtered) separately.
    Set gap_span=0 to skip this and use the raw thresholded mask as-is.

    get_motion_regions() (valid after any process() call) returns the
    bounding boxes of connected components in the current (closed) mask
    with area >= min_area. min_area and gap_span are both in small-frame
    (width x height) pixels - that's where the mask/cc analysis actually
    happen - but the returned regions themselves are scaled back up to the
    resolution of the bgr_image last passed to process(), since that's the
    image callers actually have in hand.

    Dynamic threshold: diff_thresh is a floor, not the whole story. Each
    frame's 90th-percentile small-frame pixel diff (p90_diff - the upper
    tail of the noise distribution, a closer proxy for "how large can a
    normal noise pixel get" than the mean, which just tracks its average
    magnitude) is folded into noise_floor, an IIR (exponential moving
    average) estimate of the sensor's noise level - (1-alpha)*old +
    alpha*new, with alpha derived
    each call from the actual time since the previous one (via
    time.monotonic()) so the estimate tracks a noise_window_s-second
    window regardless of frame rate: alpha = 1 - exp(-dt / noise_window_s).
    The mask is then thresholded at max(diff_thresh, noise_multiplier *
    noise_floor) - noise_multiplier default 2.0, i.e. "2x the noise floor"
    - so on a clean, well-lit feed the static diff_thresh (which is always
    a lower bound) governs, but under noisier conditions (e.g. a noisy
    sensor at night) the threshold rises to track it and false motion from
    sensor noise is suppressed. p90_diff/noise_floor/dynamic_thresh/
    effective_thresh (the actual per-frame values) are kept as attributes,
    mainly for logging/tuning.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        diff_thresh: int = DEFAULT_DIFF_THRESH,
        min_area: int = DEFAULT_MIN_AREA,
        gap_span: int = DEFAULT_GAP_SPAN,
        noise_window_s: float = DEFAULT_NOISE_WINDOW_S,
        noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
    ):
        self.width = width
        self.height = height
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.gap_span = gap_span
        self.noise_window_s = noise_window_s
        self.noise_multiplier = noise_multiplier

        self.frame = np.empty((height, width, 3), dtype=np.uint8)  # resize dst, reused every process() call
        self.prev_frame = None
        self._motion_mask = np.zeros((height, width), dtype=np.uint8)
        self._regions = []

        self.p90_diff = None            # this call's raw p90 small-frame abs-diff; None until the 2nd process() call
        self.noise_floor = None       # IIR p90-abs-diff estimate; None until the 2nd process() call
        self.dynamic_thresh = None    # noise_multiplier * noise_floor
        self.effective_thresh = None  # max(diff_thresh, dynamic_thresh), clamped to 255 - what's actually applied
        self._last_time = None

        # Kernel sized so MORPH_CLOSE bridges gaps of up to ~gap_span px
        # (dilate grows each side by the kernel radius, so radius == span).
        self._close_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_span + 1, 2 * gap_span + 1))
            if gap_span > 0 else None
        )

    def process(self, bgr_image):
        """Subsample bgr_image (full camera size, BGR) and diff it against prev_frame.

        Resizes bgr_image down to (width, height) in place into the
        preallocated small-frame buffer (nearest-neighbor interpolation) -
        this is the only per-frame work when there's no prev_frame yet to
        diff against. Once there is one, abs-diffs the two small frames,
        grayscales it, updates noise_floor from its p90 (see p90_diff) and
        derives this call's effective_thresh from that (see the
        dynamic-threshold note on this class), thresholds at
        effective_thresh, then closes small gaps (gap_span) before
        refreshing the motion mask/regions (see
        get_motion_regions/get_motion_mask) - regions are scaled up to
        bgr_image's own resolution, which need not be the same across calls
        (e.g. if the caller changes capture resolution). prev_frame is then
        updated to this call's small frame regardless, so the next call
        has something to diff against. Returns get_motion_regions().
        """
        now = time.monotonic()
        cv2.resize(bgr_image, (self.width, self.height), dst=self.frame, interpolation=cv2.INTER_NEAREST)

        if self.prev_frame is None:
            self.prev_frame = self.frame.copy()
            self._last_time = now
            return []

        diff = cv2.absdiff(self.frame, self.prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        self.p90_diff = float(np.percentile(gray_diff, 90))
        if self.noise_floor is None:
            self.noise_floor = self.p90_diff
        else:
            dt = now - self._last_time
            alpha = 1.0 - math.exp(-dt / self.noise_window_s) if self.noise_window_s > 0 else 1.0
            self.noise_floor = (1 - alpha) * self.noise_floor + alpha * self.p90_diff
        self.dynamic_thresh = self.noise_multiplier * self.noise_floor
        self.effective_thresh = min(255.0, max(self.diff_thresh, self.dynamic_thresh))

        cv2.threshold(gray_diff, self.effective_thresh, 255, cv2.THRESH_BINARY, dst=self._motion_mask)
        if self._close_kernel is not None:
            cv2.morphologyEx(self._motion_mask, cv2.MORPH_CLOSE, self._close_kernel, dst=self._motion_mask)
        image_h, image_w = bgr_image.shape[:2]
        self._regions = self._find_regions(self._motion_mask, image_w / self.width, image_h / self.height)

        self.prev_frame = self.frame.copy()
        self._last_time = now
        return self._regions

    def _find_regions(self, mask, scale_x, scale_y):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = []
        for i in range(1, num_labels):  # label 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])  # small-frame px - filtered at small-frame scale
            if area < self.min_area:
                continue
            x = round(stats[i, cv2.CC_STAT_LEFT] * scale_x)
            y = round(stats[i, cv2.CC_STAT_TOP] * scale_y)
            w = round(stats[i, cv2.CC_STAT_WIDTH] * scale_x)
            h = round(stats[i, cv2.CC_STAT_HEIGHT] * scale_y)
            regions.append((x, y, w, h, round(area * scale_x * scale_y)))
        return regions

    def get_motion_regions(self):
        """Bounding boxes of the last process() call's motion, as (x, y, w, h, area).

        In the pixel coordinates of the bgr_image passed to that process()
        call (not the small working frame). Empty if process() has not yet
        been called, or was called only once (no prev_frame yet on that
        first call).
        """
        return self._regions

    def get_motion_mask(self):
        """The last process() call's thresholded, gap-closed abs-diff mask (small-frame size, uint8 0/255)."""
        return self._motion_mask
