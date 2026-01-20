"""File operations for txttmd."""

import logging
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Encodings to try when reading files
ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

# Invalid filename characters (Windows)
INVALID_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def read_note(filepath: Path) -> str:
    """
    Read note content with multi-encoding support.

    Args:
        filepath: Path to the note file.

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        UnicodeDecodeError: If file cannot be decoded with any encoding.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    last_error: Optional[Exception] = None

    for encoding in ENCODINGS:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            logger.debug(f"Successfully read {filepath} with encoding {encoding}")
            return content
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "all",
        b"",
        0,
        0,
        f"Could not decode {filepath} with any of {ENCODINGS}",
    ) from last_error


def write_markdown(filepath: Path, content: str) -> None:
    """
    Write markdown content to file using atomic write.

    Uses a temporary file and atomic rename for safety.

    Args:
        filepath: Destination path.
        content: Markdown content to write.
    """
    filepath = Path(filepath)

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then rename (atomic on most systems)
    temp_dir = filepath.parent
    fd = None
    temp_path = None

    try:
        fd, temp_path = tempfile.mkstemp(
            suffix=".md.tmp",
            prefix=".txttmd_",
            dir=temp_dir,
        )
        temp_path = Path(temp_path)

        with open(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        fd = None  # File descriptor is now closed

        # Atomic rename (on Windows, need to remove target first)
        if filepath.exists():
            filepath.unlink()
        temp_path.rename(filepath)

        logger.debug(f"Successfully wrote {filepath}")

    except Exception:
        # Clean up temp file on error
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
    finally:
        # Ensure file descriptor is closed
        if fd is not None:
            try:
                import os
                os.close(fd)
            except OSError:
                pass


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Remove invalid characters from filename.

    Args:
        filename: Original filename.
        replacement: Character to replace invalid chars with.

    Returns:
        Sanitized filename.
    """
    # Replace invalid characters
    sanitized = INVALID_CHARS_PATTERN.sub(replacement, filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")

    # Collapse multiple underscores/spaces
    sanitized = re.sub(r"[_\s]+", "_", sanitized)

    # Limit length (Windows max is 255)
    max_len = 200  # Leave room for path and extension
    if len(sanitized) > max_len:
        # Preserve extension
        name_part = sanitized[:max_len]
        sanitized = name_part.rsplit("_", 1)[0] if "_" in name_part else name_part

    # Fallback for empty result
    if not sanitized:
        sanitized = "untitled"

    return sanitized


def resolve_filename_conflict(filepath: Path) -> Path:
    """
    Resolve filename conflict by adding timestamp suffix.

    Args:
        filepath: Desired filepath that may conflict.

    Returns:
        Non-conflicting filepath (original if no conflict).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{stem}_{timestamp}{suffix}"
    new_path = parent / new_name

    # If still conflicts (rare), add counter
    counter = 1
    while new_path.exists():
        new_name = f"{stem}_{timestamp}_{counter}{suffix}"
        new_path = parent / new_name
        counter += 1

    logger.debug(f"Resolved conflict: {filepath} -> {new_path}")
    return new_path


def move_to_archive(source: Path, archive_dir: Path) -> Path:
    """
    Move file to archive directory with conflict resolution.

    Args:
        source: Source file path.
        archive_dir: Archive directory path.

    Returns:
        Final path of archived file.
    """
    source = Path(source)
    archive_dir = Path(archive_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Create archive directory structure by date
    date_dir = archive_dir / datetime.now().strftime("%Y/%m")
    date_dir.mkdir(parents=True, exist_ok=True)

    # Determine destination
    dest = date_dir / source.name
    dest = resolve_filename_conflict(dest)

    # Move file
    shutil.move(str(source), str(dest))
    logger.info(f"Archived {source.name} to {dest}")

    return dest


def ensure_category_exists(base_path: Path, category_path: str) -> Path:
    """
    Ensure category directory exists (mkdir -p equivalent).

    Args:
        base_path: Base notes directory.
        category_path: Relative category path.

    Returns:
        Full path to category directory.
    """
    base_path = Path(base_path)
    full_path = base_path / category_path

    full_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured category exists: {full_path}")

    return full_path


def get_file_extension(filepath: Path) -> str:
    """
    Get lowercase file extension.

    Args:
        filepath: Path to file.

    Returns:
        Lowercase extension including dot (e.g., ".txt").
    """
    return Path(filepath).suffix.lower()


def is_supported_file(filepath: Path, supported_extensions: list[str]) -> bool:
    """
    Check if file has a supported extension.

    Args:
        filepath: Path to file.
        supported_extensions: List of supported extensions (e.g., [".txt", ".md"]).

    Returns:
        True if file extension is supported.
    """
    ext = get_file_extension(filepath)
    return ext in [e.lower() for e in supported_extensions]


def matches_ignore_pattern(filepath: Path, patterns: list[str]) -> bool:
    """
    Check if filename matches any ignore pattern.

    Args:
        filepath: Path to file.
        patterns: List of glob-like patterns (e.g., [".*", "_*"]).

    Returns:
        True if filename matches any pattern.
    """
    filename = Path(filepath).name

    for pattern in patterns:
        # Convert simple glob to regex
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        if re.match(f"^{regex_pattern}$", filename):
            return True

    return False


def get_note_stats(content: str) -> dict:
    """
    Get statistics about note content.

    Args:
        content: Note content.

    Returns:
        Dictionary with word_count, line_count, char_count, has_code_blocks.
    """
    lines = content.split("\n")
    words = content.split()

    return {
        "word_count": len(words),
        "line_count": len(lines),
        "char_count": len(content),
        "has_code_blocks": "```" in content or any(line.startswith("    ") for line in lines),
    }


def ensure_directories(folders_config) -> None:
    """
    Ensure all required directories exist.

    Args:
        folders_config: FolderConfig instance with inbox, output, archive paths.
    """
    folders_config.inbox.mkdir(parents=True, exist_ok=True)
    folders_config.output.mkdir(parents=True, exist_ok=True)
    folders_config.archive.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ensured directories exist: inbox={folders_config.inbox}, output={folders_config.output}, archive={folders_config.archive}")
