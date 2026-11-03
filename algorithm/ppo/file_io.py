from pathlib import Path
import sys
import time


def _replace_with_retry(source, destination):
    for attempt in range(8):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1.0))


def replace_file(source, destination, fallback=None):
    """Commit a complete file, tolerating temporary Windows sharing conflicts."""
    source, destination = Path(source), Path(destination)
    try:
        _replace_with_retry(source, destination)
        return destination
    except PermissionError:
        if fallback is None:
            raise
        fallback = Path(fallback)
        _replace_with_retry(source, fallback)
        print(
            f"Save target remains inaccessible: {destination}. "
            f"Saved the complete file to: {fallback}",
            file=sys.stderr,
            flush=True,
        )
        return fallback
