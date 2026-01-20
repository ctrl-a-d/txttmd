"""Reconfiguration menu for txttmd."""

from pathlib import Path
from typing import Optional

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import Config, ConfigLoader, Category, ProviderConfig
from src.setup.wizard import PROVIDERS, WIZARD_STYLE, _add_custom_provider


def run_reconfigure_menu() -> Optional[Config]:
    """
    Main reconfiguration menu entry point.

    Returns:
        Modified Config if saved, None if cancelled or no config exists.
    """
    console = Console()
    config_path = Path("config/config.yaml")

    # Check if config exists
    if not config_path.exists():
        console.print("[red]No configuration found. Run setup wizard first:[/red]")
        console.print("  [cyan]python -m src.setup.wizard[/cyan]")
        return None

    # Load existing config
    config = ConfigLoader.load(config_path)

    console.print(Panel.fit(
        "Modify your txttmd configuration",
        title="Reconfigure",
        border_style="cyan"
    ))

    while True:
        action = questionary.select(
            "What would you like to change?",
            choices=[
                Choice(title="Providers - Add, remove, or modify LLM providers", value="providers"),
                Choice(title="Categories - Manage note categories", value="categories"),
                Choice(title="Processing - Confidence threshold, extensions", value="processing"),
                Choice(title="API Keys - Update provider API keys", value="apikeys"),
                Choice(title="View Current Config", value="view"),
                Choice(title="Done - Save and exit", value="done"),
            ],
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "done":
            break
        elif action == "providers":
            _reconfigure_providers_menu(config, console)
        elif action == "categories":
            _reconfigure_categories_menu(config, console)
        elif action == "processing":
            _reconfigure_processing_menu(config, console)
        elif action == "apikeys":
            _reconfigure_apikeys_menu(config, console)
        elif action == "view":
            _show_current_config(config, console)

    # Save changes
    save = questionary.confirm(
        "Save changes to config?",
        default=True,
        style=WIZARD_STYLE
    ).ask()

    if save:
        ConfigLoader.save(config, config_path)
        console.print("[green]Configuration saved.[/green]")
        return config
    else:
        console.print("[yellow]Changes discarded.[/yellow]")
        return None


def _reconfigure_providers_menu(config: Config, console: Console) -> None:
    """Manage LLM providers."""
    while True:
        # Build choices showing current providers
        choices = []
        for name, prov in config.llm.providers.items():
            status = "enabled" if prov.enabled else "disabled"
            privacy = _get_provider_privacy(name)
            choices.append(Choice(
                title=f"[{status}] {name} ({prov.model}) [{privacy}]",
                value=f"edit:{name}"
            ))

        choices.extend([
            Choice(title="+ Add built-in provider", value="add_builtin"),
            Choice(title="+ Add custom provider", value="add_custom"),
            Choice(title="- Remove provider", value="remove"),
            Choice(title="Toggle provider on/off", value="toggle"),
            Choice(title="Back", value="back"),
        ])

        action = questionary.select(
            "Providers:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "back":
            break
        elif action == "add_builtin":
            _add_builtin_provider(config, console)
        elif action == "add_custom":
            custom = _add_custom_provider(console)
            if custom:
                _add_custom_to_config(config, custom, console)
        elif action == "remove":
            _remove_provider(config, console)
        elif action == "toggle":
            _toggle_provider(config, console)
        elif action.startswith("edit:"):
            provider_name = action.split(":", 1)[1]
            _edit_provider(config, provider_name, console)


def _get_provider_privacy(name: str) -> str:
    """Get privacy level for a provider."""
    if name in PROVIDERS:
        return PROVIDERS[name].get("privacy", "EXTERNAL")
    return "EXTERNAL"


def _add_builtin_provider(config: Config, console: Console) -> None:
    """Add a built-in provider to config."""
    current = list(config.llm.providers.keys())
    available = [k for k in PROVIDERS.keys() if k not in current]

    if not available:
        console.print("[yellow]All built-in providers already configured.[/yellow]")
        return

    # Build choices for available providers
    provider_choices = [
        Choice(
            title=(
                f"{PROVIDERS[key]['name']} "
                f"[{PROVIDERS[key]['type']}] "
                f"[{PROVIDERS[key].get('privacy', 'EXTERNAL')}] "
                f"- {PROVIDERS[key]['description']}"
            ),
            value=key
        )
        for key in available
    ]
    provider_choices.append(Choice(title="Cancel", value=None))

    selected = questionary.select(
        "Select provider to add:",
        choices=provider_choices,
        style=WIZARD_STYLE
    ).ask()

    if not selected:
        return

    info = PROVIDERS[selected]

    # Prompt for API key if needed
    api_key = None
    if info["env_key"]:
        api_key = questionary.password(
            f"Enter {info['env_key']} (leave empty to set later):",
            style=WIZARD_STYLE
        ).ask()

    # Create provider config
    pconfig = ProviderConfig(
        name=selected,
        model=info["default_model"],
        api_key=api_key if api_key else None,
        base_url=info.get("base_url"),
        enabled=True,
        timeout=60,
        max_retries=3,
    )

    config.llm.providers[selected] = pconfig

    # Update .env if API key provided
    if api_key:
        _update_env_key(info["env_key"], api_key)

    console.print(f"[green]Added provider '{selected}'[/green]")


def _add_custom_to_config(config: Config, custom: dict, console: Console) -> None:
    """Add a custom provider to config."""
    pconfig = ProviderConfig(
        name=custom["id"],
        model=custom["model"],
        api_key=custom.get("api_key"),
        base_url=custom["base_url"],
        enabled=True,
        timeout=60,
        max_retries=3,
    )

    config.llm.providers[custom["id"]] = pconfig

    # Update .env if API key provided
    if custom.get("api_key") and custom.get("env_key"):
        _update_env_key(custom["env_key"], custom["api_key"])

    console.print(f"[green]Added custom provider '{custom['id']}'[/green]")


def _remove_provider(config: Config, console: Console) -> None:
    """Remove a provider from config."""
    if not config.llm.providers:
        console.print("[yellow]No providers configured.[/yellow]")
        return

    choices = list(config.llm.providers.keys())
    choices.append("Cancel")

    to_remove = questionary.select(
        "Remove which provider?",
        choices=choices,
        style=WIZARD_STYLE
    ).ask()

    if to_remove and to_remove != "Cancel":
        del config.llm.providers[to_remove]

        # Update default if removed
        if config.llm.default_provider == to_remove:
            if config.llm.providers:
                config.llm.default_provider = list(config.llm.providers.keys())[0]
            else:
                config.llm.default_provider = ""

        console.print(f"[green]Removed provider '{to_remove}'[/green]")


def _toggle_provider(config: Config, console: Console) -> None:
    """Toggle a provider's enabled state."""
    if not config.llm.providers:
        console.print("[yellow]No providers configured.[/yellow]")
        return

    choices = [
        f"{'[enabled]' if p.enabled else '[disabled]'} {name}"
        for name, p in config.llm.providers.items()
    ]
    choices.append("Cancel")

    selected = questionary.select(
        "Toggle which provider?",
        choices=choices,
        style=WIZARD_STYLE
    ).ask()

    if selected and selected != "Cancel":
        # Extract provider name
        name = selected.split("] ", 1)[1]
        config.llm.providers[name].enabled = not config.llm.providers[name].enabled
        new_state = "enabled" if config.llm.providers[name].enabled else "disabled"
        console.print(f"[green]Provider '{name}' is now {new_state}[/green]")


def _edit_provider(config: Config, provider_name: str, console: Console) -> None:
    """Edit a specific provider's settings."""
    prov = config.llm.providers.get(provider_name)
    if not prov:
        return

    while True:
        choices = [
            Choice(title=f"Model: {prov.model}", value="model"),
            Choice(title=f"Enabled: {prov.enabled}", value="enabled"),
            Choice(title=f"Timeout: {prov.timeout}s", value="timeout"),
            Choice(title=f"Max retries: {prov.max_retries}", value="retries"),
            Choice(title="Back", value="back"),
        ]

        if prov.base_url:
            choices.insert(1, Choice(title=f"Base URL: {prov.base_url}", value="base_url"))

        action = questionary.select(
            f"Edit {provider_name}:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "back":
            break
        elif action == "model":
            new_model = questionary.text(
                "New model name:",
                default=prov.model,
                style=WIZARD_STYLE
            ).ask()
            if new_model:
                prov.model = new_model
        elif action == "enabled":
            prov.enabled = not prov.enabled
            console.print(f"[green]Provider {'enabled' if prov.enabled else 'disabled'}[/green]")
        elif action == "timeout":
            new_timeout = questionary.text(
                "Timeout (seconds):",
                default=str(prov.timeout),
                style=WIZARD_STYLE
            ).ask()
            try:
                prov.timeout = int(new_timeout)
            except ValueError:
                console.print("[red]Invalid number[/red]")
        elif action == "retries":
            new_retries = questionary.text(
                "Max retries:",
                default=str(prov.max_retries),
                style=WIZARD_STYLE
            ).ask()
            try:
                prov.max_retries = int(new_retries)
            except ValueError:
                console.print("[red]Invalid number[/red]")
        elif action == "base_url":
            new_url = questionary.text(
                "Base URL:",
                default=prov.base_url or "",
                style=WIZARD_STYLE
            ).ask()
            if new_url:
                prov.base_url = new_url


def _reconfigure_categories_menu(config: Config, console: Console) -> None:
    """Manage categories - similar to wizard pattern."""
    while True:
        # Show current categories
        choices = [f"{c.name} ({c.path})" for c in config.categories]
        choices.append("+ Add new category")
        choices.append("- Remove category")
        choices.append("Edit category")
        choices.append("Back")

        action = questionary.select(
            "Categories:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "Back":
            break
        elif action == "+ Add new category":
            name = questionary.text("Category name:", style=WIZARD_STYLE).ask()
            if name:
                # Check for duplicates
                if config.get_category_by_name(name):
                    console.print(f"[yellow]Category '{name}' already exists[/yellow]")
                    continue

                path = questionary.text("Folder name:", default=name, style=WIZARD_STYLE).ask()
                keywords = questionary.text("Keywords (comma-separated):", style=WIZARD_STYLE).ask()
                description = questionary.text("Description (optional):", style=WIZARD_STYLE).ask()

                config.categories.append(Category(
                    name=name,
                    path=path or name,
                    keywords=[k.strip() for k in (keywords or "").split(",") if k.strip()],
                    description=description or ""
                ))
                console.print(f"[green]Added category '{name}'[/green]")

        elif action == "- Remove category":
            if config.categories:
                to_remove = questionary.select(
                    "Remove which?",
                    choices=[c.name for c in config.categories] + ["Cancel"],
                    style=WIZARD_STYLE
                ).ask()
                if to_remove and to_remove != "Cancel":
                    config.categories = [c for c in config.categories if c.name != to_remove]
                    console.print(f"[green]Removed category '{to_remove}'[/green]")

        elif action == "Edit category":
            if config.categories:
                to_edit = questionary.select(
                    "Edit which?",
                    choices=[c.name for c in config.categories] + ["Cancel"],
                    style=WIZARD_STYLE
                ).ask()
                if to_edit and to_edit != "Cancel":
                    _edit_category(config, to_edit, console)


def _edit_category(config: Config, category_name: str, console: Console) -> None:
    """Edit a specific category."""
    cat = config.get_category_by_name(category_name)
    if not cat:
        return

    while True:
        choices = [
            Choice(title=f"Name: {cat.name}", value="name"),
            Choice(title=f"Path: {cat.path}", value="path"),
            Choice(title=f"Keywords: {', '.join(cat.keywords) or '(none)'}", value="keywords"),
            Choice(title=f"Description: {cat.description or '(none)'}", value="description"),
            Choice(title="Back", value="back"),
        ]

        action = questionary.select(
            f"Edit category '{cat.name}':",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "back":
            break
        elif action == "name":
            new_name = questionary.text("New name:", default=cat.name, style=WIZARD_STYLE).ask()
            if new_name and new_name != cat.name:
                if config.get_category_by_name(new_name):
                    console.print(f"[yellow]Category '{new_name}' already exists[/yellow]")
                else:
                    cat.name = new_name
        elif action == "path":
            new_path = questionary.text("New path:", default=cat.path, style=WIZARD_STYLE).ask()
            if new_path:
                cat.path = new_path
        elif action == "keywords":
            new_keywords = questionary.text(
                "Keywords (comma-separated):",
                default=", ".join(cat.keywords),
                style=WIZARD_STYLE
            ).ask()
            cat.keywords = [k.strip() for k in (new_keywords or "").split(",") if k.strip()]
        elif action == "description":
            new_desc = questionary.text("Description:", default=cat.description, style=WIZARD_STYLE).ask()
            cat.description = new_desc or ""


def _reconfigure_processing_menu(config: Config, console: Console) -> None:
    """Modify processing settings."""
    while True:
        choices = [
            Choice(
                title=f"Confidence threshold: {config.processing.confidence_threshold}",
                value="threshold"
            ),
            Choice(
                title=f"Debounce seconds: {config.processing.debounce_seconds}",
                value="debounce"
            ),
            Choice(
                title=f"Extensions: {', '.join(config.processing.supported_extensions)}",
                value="extensions"
            ),
            Choice(
                title=f"Ignore patterns: {', '.join(config.processing.ignore_patterns)}",
                value="ignore"
            ),
            Choice(
                title=f"Log level: {config.log_level}",
                value="log_level"
            ),
            Choice(title="Back", value="back"),
        ]

        action = questionary.select(
            "Processing settings:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "back":
            break
        elif action == "threshold":
            new_val = questionary.text(
                "Confidence threshold (0.0-1.0):",
                default=str(config.processing.confidence_threshold),
                style=WIZARD_STYLE
            ).ask()
            try:
                val = float(new_val)
                config.processing.confidence_threshold = max(0.0, min(1.0, val))
            except ValueError:
                console.print("[red]Invalid number[/red]")
        elif action == "debounce":
            new_val = questionary.text(
                "Debounce seconds:",
                default=str(config.processing.debounce_seconds),
                style=WIZARD_STYLE
            ).ask()
            try:
                config.processing.debounce_seconds = float(new_val)
            except ValueError:
                console.print("[red]Invalid number[/red]")
        elif action == "extensions":
            new_val = questionary.text(
                "Supported extensions (comma-separated, e.g. .txt,.md):",
                default=", ".join(config.processing.supported_extensions),
                style=WIZARD_STYLE
            ).ask()
            if new_val:
                config.processing.supported_extensions = [
                    e.strip() if e.strip().startswith(".") else f".{e.strip()}"
                    for e in new_val.split(",") if e.strip()
                ]
        elif action == "ignore":
            new_val = questionary.text(
                "Ignore patterns (comma-separated):",
                default=", ".join(config.processing.ignore_patterns),
                style=WIZARD_STYLE
            ).ask()
            if new_val:
                config.processing.ignore_patterns = [p.strip() for p in new_val.split(",") if p.strip()]
        elif action == "log_level":
            new_level = questionary.select(
                "Log level:",
                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                style=WIZARD_STYLE
            ).ask()
            if new_level:
                config.log_level = new_level


def _reconfigure_apikeys_menu(config: Config, console: Console) -> None:
    """Update API keys in .env file."""
    env_path = Path(".env")

    # Read current .env
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value

    while True:
        # Build choices showing which providers need keys
        choices = []
        for name, prov in config.llm.providers.items():
            if name in PROVIDERS and PROVIDERS[name]["env_key"]:
                env_key = PROVIDERS[name]["env_key"]
                has_key = env_key in env_vars and env_vars[env_key]
                status = "[set]" if has_key else "[not set]"
                choices.append(Choice(
                    title=f"{status} {env_key} ({name})",
                    value=env_key
                ))
            elif name not in PROVIDERS:
                # Custom provider - check for common key pattern
                env_key = f"{name.upper()}_API_KEY"
                has_key = env_key in env_vars and env_vars[env_key]
                status = "[set]" if has_key else "[not set]"
                choices.append(Choice(
                    title=f"{status} {env_key} ({name})",
                    value=env_key
                ))

        choices.append(Choice(title="Back", value="back"))

        action = questionary.select(
            "API Keys:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "back":
            break
        else:
            current = env_vars.get(action, "")
            masked = "****" + current[-4:] if len(current) > 4 else "(empty)"
            console.print(f"Current value: {masked}")

            new_key = questionary.password(
                f"New value for {action} (empty to keep current):",
                style=WIZARD_STYLE
            ).ask()

            if new_key:
                env_vars[action] = new_key
                _write_env_file(env_vars)
                console.print(f"[green]Updated {action}[/green]")


def _update_env_key(key: str, value: str) -> None:
    """Update a single key in .env file."""
    env_path = Path(".env")

    # Read current .env
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k] = v

    env_vars[key] = value
    _write_env_file(env_vars)


def _write_env_file(env_vars: dict) -> None:
    """Write env vars to .env file."""
    env_path = Path(".env")
    lines = ["# txttmd environment variables", ""]

    for key, value in env_vars.items():
        lines.append(f"{key}={value}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _show_current_config(config: Config, console: Console) -> None:
    """Display current configuration."""
    console.print()

    # Paths
    console.print(Panel.fit("Paths", border_style="cyan"))
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column(style="green")
    table.add_row("Notes path:", str(config.folders.inbox.parent))
    table.add_row("Inbox:", str(config.folders.inbox))
    table.add_row("Output:", str(config.folders.output))
    table.add_row("Archive:", str(config.folders.archive))
    console.print(table)
    console.print()

    # Providers
    console.print(Panel.fit("Providers", border_style="cyan"))
    prov_table = Table()
    prov_table.add_column("Name", style="cyan")
    prov_table.add_column("Model", style="green")
    prov_table.add_column("Status", style="yellow")
    prov_table.add_column("Privacy")

    for name, prov in config.llm.providers.items():
        privacy = _get_provider_privacy(name)
        privacy_color = "green" if privacy == "LOCAL" else "yellow"
        prov_table.add_row(
            name,
            prov.model,
            "enabled" if prov.enabled else "disabled",
            f"[{privacy_color}]{privacy}[/{privacy_color}]"
        )
    console.print(prov_table)
    console.print(f"Default provider: {config.llm.default_provider}")
    console.print()

    # Categories
    console.print(Panel.fit("Categories", border_style="cyan"))
    cat_table = Table()
    cat_table.add_column("Name", style="cyan")
    cat_table.add_column("Path", style="green")
    cat_table.add_column("Keywords", style="yellow")

    for cat in config.categories:
        keywords = ", ".join(cat.keywords[:3])
        if len(cat.keywords) > 3:
            keywords += "..."
        cat_table.add_row(cat.name, cat.path, keywords)
    console.print(cat_table)
    console.print()

    # Processing
    console.print(Panel.fit("Processing", border_style="cyan"))
    proc_table = Table(show_header=False, box=None)
    proc_table.add_column(style="cyan")
    proc_table.add_column(style="green")
    proc_table.add_row("Confidence threshold:", str(config.processing.confidence_threshold))
    proc_table.add_row("Debounce seconds:", str(config.processing.debounce_seconds))
    proc_table.add_row("Extensions:", ", ".join(config.processing.supported_extensions))
    proc_table.add_row("Log level:", config.log_level)
    console.print(proc_table)
    console.print()


if __name__ == "__main__":
    run_reconfigure_menu()
