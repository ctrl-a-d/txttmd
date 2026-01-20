"""File monitoring for txttmd using watchdog."""

import logging
from pathlib import Path
from threading import Timer
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .file_handler import is_supported_file, matches_ignore_pattern

logger = logging.getLogger(__name__)


class DebouncedNoteHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing.

    Debouncing prevents multiple rapid events for the same file
    (e.g., during file write operations) from triggering multiple
    processing attempts.
    """

    def __init__(
        self,
        callback: Callable[[Path], None],
        supported_extensions: list[str],
        ignore_patterns: list[str],
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize handler.

        Args:
            callback: Function to call when a file should be processed.
            supported_extensions: List of file extensions to process.
            ignore_patterns: List of filename patterns to ignore.
            debounce_seconds: Seconds to wait before processing.
        """
        super().__init__()
        self.callback = callback
        self.supported_extensions = supported_extensions
        self.ignore_patterns = ignore_patterns
        self.debounce_seconds = debounce_seconds
        self._timers: dict[str, Timer] = {}

    def _should_process(self, path: Path) -> bool:
        """
        Check if a file should be processed.

        Args:
            path: Path to the file.

        Returns:
            True if file should be processed.
        """
        # Must be a file
        if not path.is_file():
            return False

        # Must have supported extension
        if not is_supported_file(path, self.supported_extensions):
            logger.debug(f"Ignoring unsupported file: {path}")
            return False

        # Must not match ignore patterns
        if matches_ignore_pattern(path, self.ignore_patterns):
            logger.debug(f"Ignoring file matching ignore pattern: {path}")
            return False

        return True

    def _schedule_processing(self, filepath: Path) -> None:
        """
        Schedule file processing with debouncing.

        Args:
            filepath: Path to the file.
        """
        key = str(filepath)

        # Cancel any existing timer for this file
        if key in self._timers:
            self._timers[key].cancel()

        def process():
            """Execute the callback and clean up."""
            del self._timers[key]
            if filepath.exists():  # File may have been deleted
                logger.info(f"Processing file: {filepath}")
                try:
                    self.callback(filepath)
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")

        # Schedule new timer
        timer = Timer(self.debounce_seconds, process)
        self._timers[key] = timer
        timer.start()
        logger.debug(f"Scheduled processing for {filepath} in {self.debounce_seconds}s")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._should_process(path):
            self._schedule_processing(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._should_process(path):
            self._schedule_processing(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events (file moved into watched directory)."""
        if event.is_directory:
            return

        # Check if destination is in watched directory
        dest_path = Path(event.dest_path)
        if self._should_process(dest_path):
            self._schedule_processing(dest_path)

    def cancel_all_pending(self) -> None:
        """Cancel all pending processing timers."""
        for key, timer in list(self._timers.items()):
            timer.cancel()
        self._timers.clear()
        logger.debug("Cancelled all pending processing timers")


class NoteFileMonitor:
    """
    Monitor a directory for new/modified note files.

    Wraps watchdog Observer with debounced event handling.
    """

    def __init__(
        self,
        watch_path: Path,
        callback: Callable[[Path], None],
        supported_extensions: list[str],
        ignore_patterns: list[str],
        debounce_seconds: float = 2.0,
    ):
        """
        Initialize monitor.

        Args:
            watch_path: Directory to monitor.
            callback: Function to call when a file should be processed.
            supported_extensions: List of file extensions to process.
            ignore_patterns: List of filename patterns to ignore.
            debounce_seconds: Seconds to wait before processing.
        """
        self.watch_path = Path(watch_path)
        self._observer: Optional[Observer] = None

        self._handler = DebouncedNoteHandler(
            callback=callback,
            supported_extensions=supported_extensions,
            ignore_patterns=ignore_patterns,
            debounce_seconds=debounce_seconds,
        )

    def start(self) -> None:
        """Start monitoring the directory."""
        if self._observer is not None:
            logger.warning("Monitor already running")
            return

        # Ensure directory exists
        self.watch_path.mkdir(parents=True, exist_ok=True)

        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.watch_path),
            recursive=False,
        )
        self._observer.start()
        logger.info(f"Started monitoring: {self.watch_path}")

    def stop(self) -> None:
        """Stop monitoring and clean up."""
        if self._observer is None:
            return

        # Cancel pending timers
        self._handler.cancel_all_pending()

        # Stop observer
        self._observer.stop()
        self._observer.join(timeout=5)
        self._observer = None
        logger.info("Stopped monitoring")

    def is_alive(self) -> bool:
        """Check if monitor is running."""
        return self._observer is not None and self._observer.is_alive()

    def process_existing_files(self) -> int:
        """
        Process any existing files in the watch directory.

        Returns:
            Number of files queued for processing.
        """
        count = 0
        for path in self.watch_path.iterdir():
            if self._handler._should_process(path):
                self._handler._schedule_processing(path)
                count += 1

        if count > 0:
            logger.info(f"Queued {count} existing files for processing")

        return count
