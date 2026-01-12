import time, os, sys, threading
try:
    import psutil
except ImportError:
    raise SystemExit("Please `pip install psutil` to measure total process memory accurately.")

# --- OS-level peak RSS helper (best-effort cross-platform) ---
def _ru_maxrss_bytes_or_none():
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KiB.
        if sys.platform == "darwin":
            return int(ru)
        else:
            return int(ru) * 1024
    except Exception:
        return None  # Not available on Windows; sampling will still work.

def _format_bytes(n):
    for unit in ["B","KiB","MiB","GiB","TiB","PiB"]:
        if n < 1024 or unit == "PiB":
            return f"{n:.2f} {unit}"
        n /= 1024

class PeakMemory:
    """
    Context manager to measure peak RSS during a code block (single run).
    Combines fast sampling of psutil RSS with OS ru_maxrss for extra accuracy.
    """
    def __init__(self, sample_interval_sec=0.005):
        self.sample_interval_sec = sample_interval_sec
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.entry_rss = 0
        self.exit_rss  = 0
        self.peak_rss_sampled = 0
        self.ru_start = None
        self.ru_end   = None

    def _sampler(self):
        local_peak = self.entry_rss
        sleep = time.sleep
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > local_peak:
                    local_peak = rss
            except Exception:
                pass
            sleep(self.sample_interval_sec)
        self.peak_rss_sampled = max(self.peak_rss_sampled, local_peak)

    def __enter__(self):
        # Record entry stats and start sampler thread
        self.entry_rss = self.proc.memory_info().rss
        self.ru_start = _ru_maxrss_bytes_or_none()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Stop sampler and record exit/OS-peak
        self.exit_rss = self.proc.memory_info().rss
        self._stop.set()
        self._thread.join()
        self.ru_end = _ru_maxrss_bytes_or_none()

    def results(self):
        # Window peak = max(sampled window peak, any OS-level new peak seen during window)
        os_window_peak = None
        if self.ru_start is not None and self.ru_end is not None and self.ru_end > self.ru_start:
            os_window_peak = self.ru_end  # OS observed a higher lifetime peak during this window

        peak_during_block = max(self.peak_rss_sampled, os_window_peak or 0)
        added_peak = peak_during_block - self.entry_rss if peak_during_block else 0

        return {
            "entry_rss_bytes": self.entry_rss,
            "exit_rss_bytes": self.exit_rss,
            "peak_rss_during_block_bytes": peak_during_block or self.peak_rss_sampled or self.exit_rss,
            "added_peak_over_entry_bytes": max(0, added_peak),
            "ru_maxrss_start_bytes": self.ru_start,
            "ru_maxrss_end_bytes": self.ru_end,
            "human_readable": {
                "entry_rss": _format_bytes(self.entry_rss),
                "exit_rss": _format_bytes(self.exit_rss),
                "peak_rss_during_block": _format_bytes(peak_during_block or self.peak_rss_sampled or self.exit_rss),
                "added_peak_over_entry": _format_bytes(max(0, added_peak)),
            }
        }