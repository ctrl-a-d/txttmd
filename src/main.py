"""Main orchestration module for txttmd."""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config import Config, ConfigLoader
from .file_handler import (
    ensure_category_exists,
    ensure_directories,
    get_file_extension,
    move_to_archive,
    read_note,
    resolve_filename_conflict,
    sanitize_filename,
    write_markdown,
)
from .file_monitor import NoteFileMonitor
from .note_processor import NoteProcessor, NoteResult

console = Console()
logger = logging.getLogger("txttmd")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Rich handler for console output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    rich_handler.setLevel(log_level)

    handlers: list[logging.Handler] = [rich_handler]

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=handlers,
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("src.file_monitor").setLevel(logging.WARNING)
    logging.getLogger("src.file_handler").setLevel(logging.WARNING)
    logging.getLogger("src.note_processor").setLevel(logging.WARNING)


class NoteOrganizer:
    """Main application class that orchestrates note processing."""

    def __init__(self, config: Config):
        """
        Initialize the organizer.

        Args:
            config: Application configuration.
        """
        self.config = config
        self._running = False
        self._monitor: Optional[NoteFileMonitor] = None

        # Ensure directories exist
        ensure_directories(config.folders)

        # Initialize the LLM processor
        self.processor = NoteProcessor(
            llm_config=config.llm,
            categories=config.get_category_names(),
        )

    def process_file(self, filepath: Path) -> bool:
        """
        Process a single note file.

        Args:
            filepath: Path to the note file.

        Returns:
            True if processing succeeded.
        """
        filepath = Path(filepath)
        logger.info(f"[bold]Processing:[/bold] {filepath.name}")

        try:
            # Read the note
            content = read_note(filepath)
            file_type = get_file_extension(filepath)

            # Process through LLM
            result = self.processor.process(content, file_type)
            conf_color = "green" if result.confidence >= 0.8 else "yellow" if result.confidence >= 0.5 else "red"
            logger.info(
                f"  → [cyan]{result.category}[/cyan] "
                f"([{conf_color}]{result.confidence:.0%}[/{conf_color}]) "
                f"via [dim]{result.provider_used}[/dim]"
            )

            # Determine final category
            final_category = self._determine_category(result)

            # Prepare content (add review flag if needed)
            final_content = self._prepare_content(result)

            # Write to output
            output_path = self._write_output(result, final_category, final_content)
            logger.info(f"  → [green]Saved:[/green] {output_path.name}")

            # Archive original
            archive_path = move_to_archive(filepath, self.config.folders.archive)

            console.print("[green]✓[/green] Watching for new notes... [dim](Ctrl+C to stop)[/dim]")
            return True

        except Exception as e:
            logger.error(f"[red]Failed:[/red] {filepath.name} - {e}")
            console.print("[green]✓[/green] Watching for new notes... [dim](Ctrl+C to stop)[/dim]")
            return False

    def _determine_category(self, result: NoteResult) -> str:
        """
        Determine the final category for a note.

        Uses fallback category if confidence is below threshold.

        Args:
            result: Processing result.

        Returns:
            Category name to use.
        """
        # Check if category exists
        category = self.config.get_category_by_name(result.category)

        if category is None:
            logger.warning(f"  → [yellow]Unknown category '{result.category}', using fallback[/yellow]")
            return self.config.fallback.category

        # Check confidence threshold
        if result.confidence < self.config.processing.confidence_threshold:
            if self.config.fallback.enabled:
                logger.warning(
                    f"  → [yellow]Low confidence ({result.confidence:.0%}), using fallback[/yellow]"
                )
                return self.config.fallback.category

        return category.path

    def _prepare_content(self, result: NoteResult) -> str:
        """
        Prepare final content, adding review flag if needed.

        Args:
            result: Processing result.

        Returns:
            Final markdown content.
        """
        content = result.content

        # Add review flag for low confidence
        if result.confidence < self.config.processing.confidence_threshold:
            review_flag = self.config.fallback.review_flag
            content = f"{review_flag}\n\n{content}"

        # Add metadata footer
        metadata = [
            "",
            "---",
            f"*Processed by txttmd | Provider: {result.provider_used} | "
            f"Category: {result.category} | Confidence: {result.confidence:.2f}*",
        ]

        if result.tags:
            metadata.append(f"*Tags: {', '.join(result.tags)}*")

        return content + "\n".join(metadata)

    def _write_output(
        self,
        result: NoteResult,
        category: str,
        content: str,
    ) -> Path:
        """
        Write processed note to output directory.

        Args:
            result: Processing result.
            category: Target category path.
            content: Final content to write.

        Returns:
            Path where note was written.
        """
        # Ensure category directory exists
        category_path = ensure_category_exists(
            self.config.folders.output,
            category,
        )

        # Sanitize filename
        filename = sanitize_filename(result.filename)
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Resolve conflicts
        output_path = category_path / filename
        output_path = resolve_filename_conflict(output_path)

        # Write file
        write_markdown(output_path, content)

        return output_path

    def run(self) -> None:
        """Run the note organizer (blocking)."""
        self._running = True

        # Set up signal handlers
        self._setup_signals()

        # Start file monitor
        self._monitor = NoteFileMonitor(
            watch_path=self.config.folders.inbox,
            callback=self.process_file,
            supported_extensions=self.config.processing.supported_extensions,
            ignore_patterns=self.config.processing.ignore_patterns,
            debounce_seconds=self.config.processing.debounce_seconds,
        )
        self._monitor.start()

        # Process existing files
        self._monitor.process_existing_files()

        console.print("[green]✓[/green] Watching for new notes... [dim](Ctrl+C to stop)[/dim]")

        # Main loop (Windows compatible - no signal.pause())
        try:
            while self._running:
                if not self._monitor.is_alive():
                    console.print("[red]Error: Monitor stopped unexpectedly[/red]")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the note organizer."""
        self._running = False

        if self._monitor:
            self._monitor.stop()
            self._monitor = None

        console.print("[dim]Stopped.[/dim]")

    def _setup_signals(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            console.print("\n[yellow]Shutting down...[/yellow]")
            self._running = False

        # SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, handle_signal)

        # SIGTERM (only on Unix)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)


def main(config_path: Optional[str] = None) -> int:
    """
    Main entry point.

    Args:
        config_path: Optional path to config file.

    Returns:
        Exit code (0 for success).
    """
    try:
        # Load configuration
        cfg_path = Path(config_path) if config_path else None
        config = ConfigLoader.load(config_path=cfg_path)

        # Set up logging
        setup_logging(config.log_level, config.log_file)

        # Show startup banner
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]txttmd[/bold cyan] [dim]Note Automation System[/dim]",
                border_style="cyan",
            )
        )

        # Show config table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="green")
        table.add_row("Inbox", str(config.folders.inbox))
        table.add_row("Output", str(config.folders.output))
        table.add_row("Categories", ", ".join(config.get_category_names()))
        table.add_row("Providers", ", ".join(p for p in config.llm.providers if config.llm.providers[p].enabled))
        console.print(table)
        console.print()

        # Run organizer
        organizer = NoteOrganizer(config)
        organizer.run()

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Allow config path as command line argument
    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(config_arg))
