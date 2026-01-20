"""CLI tool for managing txttmd."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Category, ConfigLoader, Config

console = Console()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration."""
    path = Path(config_path) if config_path else None
    return ConfigLoader.load(config_path=path)


def save_config(config: Config, config_path: Optional[str] = None) -> None:
    """Save configuration."""
    path = Path(config_path) if config_path else Path("config/config.yaml")
    ConfigLoader.save(config, path)


@click.group()
@click.option("--config", "-c", "config_path", help="Path to config file")
@click.pass_context
def cli(ctx: click.Context, config_path: Optional[str]) -> None:
    """txttmd - Note automation system management CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


# Category commands
@cli.group()
def category():
    """Manage note categories."""
    pass


@category.command("list")
@click.pass_context
def category_list(ctx: click.Context) -> None:
    """List all categories."""
    config = load_config(ctx.obj.get("config_path"))

    table = Table(title="Categories")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Keywords", style="yellow")
    table.add_column("Description")

    for cat in config.categories:
        table.add_row(
            cat.name,
            cat.path,
            ", ".join(cat.keywords[:3]) + ("..." if len(cat.keywords) > 3 else ""),
            cat.description or "-",
        )

    console.print(table)


@category.command("add")
@click.argument("name")
@click.option("--path", "-p", help="Folder path (defaults to name)")
@click.option("--keywords", "-k", help="Comma-separated keywords")
@click.option("--description", "-d", help="Category description")
@click.pass_context
def category_add(
    ctx: click.Context,
    name: str,
    path: Optional[str],
    keywords: Optional[str],
    description: Optional[str],
) -> None:
    """Add a new category."""
    config = load_config(ctx.obj.get("config_path"))

    # Check if category exists
    if config.get_category_by_name(name):
        console.print(f"[red]Category '{name}' already exists[/red]")
        return

    # Create category
    new_cat = Category(
        name=name,
        path=path or name,
        keywords=[k.strip() for k in (keywords or "").split(",") if k.strip()],
        description=description or "",
    )

    config.categories.append(new_cat)
    save_config(config, ctx.obj.get("config_path"))

    console.print(f"[green]Added category '{name}'[/green]")


@category.command("remove")
@click.argument("name")
@click.pass_context
def category_remove(ctx: click.Context, name: str) -> None:
    """Remove a category."""
    config = load_config(ctx.obj.get("config_path"))

    # Find category
    cat = config.get_category_by_name(name)
    if not cat:
        console.print(f"[red]Category '{name}' not found[/red]")
        return

    config.categories.remove(cat)
    save_config(config, ctx.obj.get("config_path"))

    console.print(f"[green]Removed category '{name}'[/green]")


# Config commands
@cli.group("config")
def config_cmd():
    """View and modify configuration."""
    pass


@config_cmd.command("get")
@click.argument("key", required=False)
@click.pass_context
def config_get(ctx: click.Context, key: Optional[str]) -> None:
    """Get configuration value(s)."""
    config = load_config(ctx.obj.get("config_path"))

    if key is None:
        # Show all config
        table = Table(title="Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("notes_path", str(config.folders.inbox.parent))
        table.add_row("inbox", str(config.folders.inbox))
        table.add_row("output", str(config.folders.output))
        table.add_row("archive", str(config.folders.archive))
        table.add_row("log_level", config.log_level)
        table.add_row("confidence_threshold", str(config.processing.confidence_threshold))
        table.add_row("fallback_enabled", str(config.fallback.enabled))
        table.add_row("fallback_category", config.fallback.category)
        table.add_row("providers", ", ".join(config.llm.providers.keys()))

        console.print(table)
    else:
        # Get specific key
        value = None
        if key == "notes_path":
            value = str(config.folders.inbox.parent)
        elif key == "inbox":
            value = str(config.folders.inbox)
        elif key == "output":
            value = str(config.folders.output)
        elif key == "archive":
            value = str(config.folders.archive)
        elif key == "log_level":
            value = config.log_level
        elif key == "confidence_threshold":
            value = str(config.processing.confidence_threshold)
        elif key == "fallback_enabled":
            value = str(config.fallback.enabled)
        elif key == "fallback_category":
            value = config.fallback.category
        elif key == "providers":
            value = ", ".join(config.llm.providers.keys())
        else:
            console.print(f"[red]Unknown key: {key}[/red]")
            return

        console.print(f"{key}: {value}")


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set configuration value."""
    config = load_config(ctx.obj.get("config_path"))

    if key == "log_level":
        config.log_level = value.upper()
    elif key == "confidence_threshold":
        config.processing.confidence_threshold = float(value)
    elif key == "fallback_enabled":
        config.fallback.enabled = value.lower() in ("true", "1", "yes")
    elif key == "fallback_category":
        config.fallback.category = value
    else:
        console.print(f"[red]Cannot set '{key}' - use config file directly[/red]")
        return

    save_config(config, ctx.obj.get("config_path"))
    console.print(f"[green]Set {key} = {value}[/green]")


# Stats command
@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show processing statistics."""
    config = load_config(ctx.obj.get("config_path"))

    # Count files in each location
    def count_files(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for f in path.rglob("*") if f.is_file())

    inbox_count = count_files(config.folders.inbox)
    output_count = count_files(config.folders.output)
    archive_count = count_files(config.folders.archive)

    table = Table(title="Statistics")
    table.add_column("Location", style="cyan")
    table.add_column("Files", style="green", justify="right")

    table.add_row("Inbox (pending)", str(inbox_count))
    table.add_row("Output (processed)", str(output_count))
    table.add_row("Archive (originals)", str(archive_count))
    table.add_row("", "")
    table.add_row("Total", str(inbox_count + output_count + archive_count))

    console.print(table)

    # Count by category
    if config.folders.output.exists():
        cat_table = Table(title="Files by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Files", style="green", justify="right")

        for cat in config.categories:
            cat_path = config.folders.output / cat.path
            cat_count = count_files(cat_path) if cat_path.exists() else 0
            cat_table.add_row(cat.name, str(cat_count))

        # Unsorted
        unsorted_path = config.folders.output / config.fallback.category
        unsorted_count = count_files(unsorted_path) if unsorted_path.exists() else 0
        if unsorted_count > 0:
            cat_table.add_row(config.fallback.category, str(unsorted_count))

        console.print(cat_table)


# Provider commands
@cli.group()
def provider():
    """Manage LLM providers."""
    pass


@provider.command("list")
@click.pass_context
def provider_list(ctx: click.Context) -> None:
    """List configured providers."""
    config = load_config(ctx.obj.get("config_path"))

    table = Table(title="LLM Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Enabled", style="yellow")
    table.add_column("API Key", style="magenta")

    for name, prov in config.llm.providers.items():
        has_key = "Yes" if prov.api_key else "No"
        table.add_row(
            name,
            prov.model,
            "Yes" if prov.enabled else "No",
            has_key,
        )

    console.print(table)


@provider.command("test")
@click.argument("name", required=False)
@click.pass_context
def provider_test(ctx: click.Context, name: Optional[str]) -> None:
    """Test provider connectivity."""
    config = load_config(ctx.obj.get("config_path"))

    from src.note_processor import NoteProcessor

    try:
        processor = NoteProcessor(config.llm, config.get_category_names())
        results = processor.health_check_all()

        table = Table(title="Provider Health Check")
        table.add_column("Provider", style="cyan")
        table.add_column("Status")

        for prov_name, healthy in results.items():
            if name and prov_name != name:
                continue
            status = "[green]OK[/green]" if healthy else "[red]FAILED[/red]"
            table.add_row(prov_name, status)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# Run command
@cli.command()
@click.option("--once", is_flag=True, help="Process existing files and exit")
@click.pass_context
def run(ctx: click.Context, once: bool) -> None:
    """Start the note processor."""
    from src.main import main as run_main

    config_path = ctx.obj.get("config_path")

    if once:
        # Process existing files only
        config = load_config(config_path)
        from src.main import NoteOrganizer

        organizer = NoteOrganizer(config)

        for filepath in config.folders.inbox.iterdir():
            if filepath.is_file():
                console.print(f"Processing: {filepath.name}")
                success = organizer.process_file(filepath)
                status = "[green]OK[/green]" if success else "[red]FAILED[/red]"
                console.print(f"  {status}")
    else:
        # Run continuously
        sys.exit(run_main(config_path))


# Setup command
@cli.command()
def setup():
    """Run the setup wizard."""
    from src.setup.wizard import run_wizard

    run_wizard()


# Reconfigure command
@cli.command()
def reconfigure():
    """Interactive menu to reconfigure existing setup."""
    from src.setup.reconfigure import run_reconfigure_menu

    result = run_reconfigure_menu()
    if result:
        console.print("Reconfiguration complete.")


def main():
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
