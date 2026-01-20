"""CLI Setup Wizard for txttmd using questionary."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import questionary
import yaml
from questionary import Choice, Style
from rich.console import Console
from rich.panel import Panel


# Custom style for questionary prompts
WIZARD_STYLE = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:green'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
    ('separator', 'fg:gray'),
    ('instruction', 'fg:gray'),
])


ASCII_BANNER = """
████████╗██╗  ██╗████████╗████████╗███╗   ███╗██████╗
╚══██╔══╝╚██╗██╔╝╚══██╔══╝╚══██╔══╝████╗ ████║██╔══██╗
   ██║    ╚███╔╝    ██║      ██║   ██╔████╔██║██║  ██║
   ██║    ██╔██╗    ██║      ██║   ██║╚██╔╝██║██║  ██║
   ██║   ██╔╝ ██╗   ██║      ██║   ██║ ╚═╝ ██║██████╔╝
   ╚═╝   ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝     ╚═╝╚═════╝
"""


# Provider metadata
PROVIDERS = {
    "claude": {
        "name": "Claude (Anthropic)",
        "type": "PAID",
        "privacy": "EXTERNAL",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
        "description": "High-quality reasoning, best for complex notes",
    },
    "openai": {
        "name": "OpenAI GPT-4",
        "type": "PAID",
        "privacy": "EXTERNAL",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "description": "Fast and capable, good all-around choice",
    },
    "groq": {
        "name": "Groq (Llama)",
        "type": "FREE",
        "privacy": "EXTERNAL",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "description": "Free tier available, very fast inference",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "type": "LOCAL",
        "privacy": "LOCAL",
        "env_key": None,
        "default_model": "llama3.2",
        "description": "Requires Ollama installed & running locally (ollama.com)",
    },
    "openrouter": {
        "name": "OpenRouter",
        "type": "PAID",
        "privacy": "EXTERNAL",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.5-sonnet",
        "base_url": "https://openrouter.ai/api/v1",
        "description": "Access multiple models via one API",
    },
    "perplexity": {
        "name": "Perplexity",
        "type": "PAID",
        "privacy": "EXTERNAL",
        "env_key": "PERPLEXITY_API_KEY",
        "default_model": "llama-3.1-sonar-large-128k-online",
        "description": "Good for research-heavy notes",
    },
}


# Default categories
DEFAULT_CATEGORIES = [
    {"name": "Projects", "path": "Projects", "keywords": ["project", "task", "todo"], "description": "Project-related notes"},
    {"name": "Ideas", "path": "Ideas", "keywords": ["idea", "brainstorm", "concept"], "description": "Ideas and brainstorming"},
    {"name": "Research", "path": "Research", "keywords": ["research", "study", "paper"], "description": "Research notes"},
    {"name": "Personal", "path": "Personal", "keywords": ["personal", "journal", "diary"], "description": "Personal notes"},
    {"name": "Work", "path": "Work", "keywords": ["work", "meeting", "client"], "description": "Work-related notes"},
]


def _check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _build_and_start_docker(console: Console) -> bool:
    """
    Build Docker image and start container.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Build the image
        console.print("\n[bold cyan]Building Docker image...[/bold cyan]")
        build_result = subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile", "-t", "txttmd:latest", "."],
            capture_output=True,
            text=True,
            timeout=300
        )

        if build_result.returncode != 0:
            console.print(f"[red]Docker build failed:[/red]\n{build_result.stderr}")
            return False

        console.print("[green]OK[/green] Docker image built successfully")

        # Start with docker-compose
        console.print("[bold cyan]Starting container...[/bold cyan]")
        start_result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if start_result.returncode != 0:
            console.print(f"[red]Failed to start container:[/red]\n{start_result.stderr}")
            return False

        console.print("[green]OK[/green] Container started successfully")

        # Show container status
        status_result = subprocess.run(
            ["docker", "ps", "--filter", "name=txttmd", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if status_result.returncode == 0 and status_result.stdout:
            console.print("\n[bold]Container Status:[/bold]")
            console.print(status_result.stdout)

        return True

    except subprocess.TimeoutExpired:
        console.print("[red]Docker operation timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def run_wizard() -> Optional[dict]:
    """
    Run the setup wizard.

    Returns:
        Configuration dict or None if cancelled.
    """
    console = Console()

    # 1. Welcome banner
    console.print(Panel(ASCII_BANNER, title="txttmd", border_style="cyan"))
    console.print("[bold]Note Automation System[/bold]")
    console.print("Transform raw notes into organized markdown.\n")

    # 2. Notes path
    default_path = str(Path.home() / "Notes")
    notes_path = questionary.path(
        "Where are your notes located?",
        default=default_path,
        style=WIZARD_STYLE
    ).ask()

    if notes_path is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return None

    # Validate path
    path = Path(notes_path)
    if not path.is_absolute():
        console.print("[red]Please use an absolute path.[/red]")
        return None
    if not path.parent.exists():
        console.print(f"[red]Parent directory does not exist: {path.parent}[/red]")
        return None

    # 2.5. Privacy Notice
    console.print()
    console.print(Panel.fit(
        "[yellow]Privacy Notice[/yellow]\n\n"
        "txttmd sends your [bold]complete note content[/bold] to the LLM provider for processing.\n"
        "No filtering or redaction is performed before transmission.\n\n"
        "[bold]External providers[/bold] (Claude, OpenAI, Groq, etc.):\n"
        "  • Notes sent to their servers via API\n"
        "  • Subject to provider's data retention policies\n"
        "  • May not be suitable for sensitive/confidential notes\n\n"
        "[bold green]Ollama (Local)[/bold green]:\n"
        "  • Requires separate Ollama installation (ollama.com)\n"
        "  • You must have Ollama running before using this option\n"
        "  • Notes stay on your machine - never sent externally\n"
        "  • Recommended for sensitive data\n\n"
        "Choose your provider(s) based on your privacy requirements.",
        border_style="yellow",
        title="Important",
    ))
    console.print()

    # Ask user to acknowledge
    acknowledged = questionary.confirm(
        "Do you understand and accept these privacy implications?",
        default=True,
        style=WIZARD_STYLE
    ).ask()

    if not acknowledged:
        console.print("[yellow]Setup cancelled. Consider using Ollama for local-only processing.[/yellow]")
        return None

    # 3. Provider selection (checkbox)
    provider_choices = [
        Choice(
            title=(
                f"{info['name']} "
                f"[{info['type']}] "
                f"[{'LOCAL' if info.get('privacy') == 'LOCAL' else 'EXTERNAL'}] "
                f"- {info['description']}"
            ),
            value=key,
            checked=False  # No defaults - let user choose
        )
        for key, info in PROVIDERS.items()
    ]

    selected_providers = questionary.checkbox(
        "Select LLM provider(s) - one or more (space to select, enter to confirm):",
        choices=provider_choices,
        style=WIZARD_STYLE
    ).ask()

    if selected_providers is None or len(selected_providers) == 0:
        console.print("[yellow]No providers selected. Setup cancelled.[/yellow]")
        return None

    # 4. API keys (only for selected providers that need them)
    api_keys = {}
    for provider in selected_providers:
        env_key = PROVIDERS[provider]["env_key"]
        if env_key:
            key = questionary.password(
                f"Enter {env_key}:",
                style=WIZARD_STYLE
            ).ask()
            if key:
                api_keys[provider] = key

    # 4.5. Custom providers
    custom_providers = []
    add_custom = questionary.confirm(
        "Add custom LLM provider? (for self-hosted or other APIs)",
        default=False,
        style=WIZARD_STYLE
    ).ask()

    while add_custom:
        custom = _add_custom_provider(console)
        if custom:
            custom_providers.append(custom)
            selected_providers.append(custom["id"])
            if custom["api_key"]:
                api_keys[custom["id"]] = custom["api_key"]

            # Ask if they want to add another
            add_custom = questionary.confirm(
                "Add another custom provider?",
                default=False,
                style=WIZARD_STYLE
            ).ask()
        else:
            add_custom = False

    # 5. Routing strategy (only if multiple providers)
    use_auto = False
    if len(selected_providers) > 1:
        use_auto = questionary.confirm(
            "Use auto-routing between providers? (recommended)",
            default=True,
            style=WIZARD_STYLE
        ).ask()

        if use_auto is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return None

    # Generate routing rules
    routing_rules = _generate_routing_rules(selected_providers, use_auto)

    # 6. Categories
    edit_categories = questionary.confirm(
        "Edit default categories?",
        default=False,
        style=WIZARD_STYLE
    ).ask()

    categories = DEFAULT_CATEGORIES.copy()
    if edit_categories:
        categories = _edit_categories_menu(categories)

    # 7. Settings
    threshold_str = questionary.text(
        "Confidence threshold (0.0-1.0):",
        default="0.7",
        style=WIZARD_STYLE
    ).ask()

    try:
        threshold = float(threshold_str) if threshold_str else 0.7
        threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        threshold = 0.7

    # Check Docker availability
    docker_available = _check_docker_available()
    use_docker = False

    if docker_available:
        use_docker = questionary.confirm(
            "Build and start Docker container?",
            default=True,
            style=WIZARD_STYLE
        ).ask()
        if use_docker is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return None
    else:
        console.print("[yellow]Docker not available - will run locally[/yellow]")

    # 8. Execute setup
    console.print("\n[bold cyan]Setting up...[/bold cyan]")

    notes_path_obj = Path(notes_path)

    console.print("Creating directories...", end=" ")
    _create_directories(notes_path_obj, categories)
    console.print("[green]done[/green]")

    console.print("Writing config.yaml...", end=" ")
    _create_config(notes_path_obj, selected_providers, categories, threshold, routing_rules, custom_providers)
    console.print("[green]done[/green]")

    console.print("Writing .env...", end=" ")
    _create_env_file(notes_path_obj, api_keys, custom_providers)
    console.print("[green]done[/green]")

    # Docker setup if requested
    docker_success = False
    if use_docker:
        docker_success = _build_and_start_docker(console)

    # Show completion message
    console.print("\n[bold green]OK Setup complete![/bold green]")

    if docker_success:
        console.print("\n[bold]txttmd is now running in Docker![/bold]")
        console.print("\nUseful commands:")
        console.print("  View logs:    [cyan]docker logs -f txttmd[/cyan]")
        console.print("  Stop:         [cyan]docker-compose -f docker/docker-compose.yml stop[/cyan]")
        console.print("  Restart:      [cyan]docker-compose -f docker/docker-compose.yml restart[/cyan]")
        console.print("  Remove:       [cyan]docker-compose -f docker/docker-compose.yml down[/cyan]")
    elif use_docker:
        console.print("\n[yellow]Docker setup failed. You can start manually:[/yellow]")
        console.print("  [cyan]cd docker && docker-compose up -d[/cyan]")
        console.print("\nOr run locally:")
        console.print("  [cyan]python -m src.main[/cyan]")
    else:
        console.print("\nTo start txttmd:")
        console.print("  [cyan]python -m src.main[/cyan]")
        console.print("\nOr use Docker:")
        console.print("  [cyan]cd docker && docker-compose up -d[/cyan]")

    # Show privacy reminder
    console.print("\n[bold]Privacy Reminder:[/bold]")

    # Create custom provider lookup for privacy check
    custom_lookup = {cp["id"]: cp for cp in custom_providers}

    external_providers = []
    local_providers = []

    for p in selected_providers:
        if p in custom_lookup:
            # Custom provider
            if custom_lookup[p]["privacy"] == "LOCAL":
                local_providers.append(p)
            else:
                external_providers.append(p)
        else:
            # Built-in provider
            if PROVIDERS[p].get("privacy") == "LOCAL":
                local_providers.append(p)
            else:
                external_providers.append(p)

    if external_providers:
        console.print(
            f"  [yellow]EXTERNAL:[/yellow] {', '.join(external_providers)}"
        )
        console.print("      → Notes will be sent to external APIs")
    if local_providers:
        console.print(
            f"  [green]LOCAL:[/green] {', '.join(local_providers)}"
        )
        console.print("      → Notes stay on your machine")
        if "ollama" in local_providers:
            console.print("      → [yellow]Ensure Ollama is installed and running:[/yellow] ollama serve")

    # Reconfigure hint
    console.print("\n[bold]To reconfigure later, run:[/bold]")
    console.print("  [cyan]python -m src.cli.notectl reconfigure[/cyan]")

    return {
        "notes_path": str(notes_path_obj),
        "providers": selected_providers,
        "categories": categories,
        "threshold": threshold,
        "use_docker": use_docker,
        "docker_success": docker_success,
    }


def _generate_routing_rules(providers: list[str], auto: bool) -> list[dict]:
    """Generate routing rules based on selected providers."""
    rules = []

    if auto:
        # Short notes -> fast provider (groq or ollama)
        fast_providers = [p for p in ["groq", "ollama"] if p in providers]
        if fast_providers:
            rules.append({
                "provider": fast_providers[0],
                "priority": 100,
                "conditions": [{"type": "word_count", "value": 500, "operator": "<"}],
            })

        # Code blocks -> best code model (claude or openai)
        code_providers = [p for p in ["claude", "openai"] if p in providers]
        if code_providers:
            rules.append({
                "provider": code_providers[0],
                "priority": 90,
                "conditions": [{"type": "contains_code_blocks"}],
            })

    # Default -> most capable
    if "claude" in providers:
        default = "claude"
    elif "openai" in providers:
        default = "openai"
    else:
        default = providers[0] if providers else "ollama"

    rules.append({
        "provider": default,
        "priority": 0,
        "conditions": [{"type": "always"}],
    })

    return rules


def _edit_categories_menu(categories: list[dict]) -> list[dict]:
    """Interactive category editor."""
    while True:
        # Show current categories
        choices = [f"{c['name']} ({c['path']})" for c in categories]
        choices.append("+ Add new category")
        choices.append("- Remove category")
        choices.append("Done")

        action = questionary.select(
            "Categories:",
            choices=choices,
            style=WIZARD_STYLE
        ).ask()

        if action is None or action == "Done":
            break
        elif action == "+ Add new category":
            name = questionary.text("Category name:", style=WIZARD_STYLE).ask()
            if name:
                path = questionary.text("Folder name:", default=name, style=WIZARD_STYLE).ask()
                keywords = questionary.text("Keywords (comma-separated):", style=WIZARD_STYLE).ask()
                categories.append({
                    "name": name,
                    "path": path or name,
                    "keywords": [k.strip() for k in (keywords or "").split(",") if k.strip()],
                    "description": ""
                })
        elif action == "- Remove category":
            if categories:
                to_remove = questionary.select(
                    "Remove which?",
                    choices=[c["name"] for c in categories],
                    style=WIZARD_STYLE
                ).ask()
                if to_remove:
                    categories = [c for c in categories if c["name"] != to_remove]

    return categories


def _add_custom_provider(console: Console) -> Optional[dict]:
    """
    Interactive prompt to add a custom LLM provider.

    Returns:
        Dict with provider config or None if cancelled.
    """
    console.print("\n[bold cyan]Add Custom Provider[/bold cyan]")
    console.print("Configure your own LLM endpoint (OpenAI-compatible API)\n")

    # Provider identifier (used in config)
    provider_id = questionary.text(
        "Provider identifier (lowercase, no spaces):",
        validate=lambda text: len(text) > 0 and text.islower() and " " not in text,
        style=WIZARD_STYLE
    ).ask()

    if not provider_id:
        return None

    # Display name
    display_name = questionary.text(
        "Display name:",
        default=provider_id.title(),
        style=WIZARD_STYLE
    ).ask()

    if not display_name:
        return None

    # Base URL
    base_url = questionary.text(
        "API base URL (e.g., http://localhost:1234/v1):",
        validate=lambda text: text.startswith("http://") or text.startswith("https://"),
        style=WIZARD_STYLE
    ).ask()

    if not base_url:
        return None

    # Model name
    model = questionary.text(
        "Model name:",
        default="default",
        style=WIZARD_STYLE
    ).ask()

    if not model:
        return None

    # Privacy level
    privacy = questionary.select(
        "Privacy level:",
        choices=[
            Choice(title="LOCAL - Runs on your machine/network", value="LOCAL"),
            Choice(title="EXTERNAL - Third-party hosted", value="EXTERNAL")
        ],
        style=WIZARD_STYLE
    ).ask()

    if not privacy:
        return None

    # API key requirement
    needs_api_key = questionary.confirm(
        "Does this provider require an API key?",
        default=False,
        style=WIZARD_STYLE
    ).ask()

    env_key = None
    api_key = None

    if needs_api_key:
        env_key = questionary.text(
            "Environment variable name for API key:",
            default=f"{provider_id.upper()}_API_KEY",
            style=WIZARD_STYLE
        ).ask()

        if env_key:
            api_key = questionary.password(
                f"Enter {env_key}:",
                style=WIZARD_STYLE
            ).ask()

    console.print(f"[green]✓[/green] Custom provider '{display_name}' configured")

    return {
        "id": provider_id,
        "name": display_name,
        "base_url": base_url,
        "model": model,
        "privacy": privacy,
        "env_key": env_key,
        "api_key": api_key,
    }


def _create_directories(notes_path: Path, categories: list[dict]) -> None:
    """Create required directories."""
    (notes_path / "_Inbox").mkdir(parents=True, exist_ok=True)
    (notes_path / "Notes").mkdir(parents=True, exist_ok=True)
    (notes_path / "_Archive").mkdir(parents=True, exist_ok=True)

    # Create category directories
    for cat in categories:
        (notes_path / "Notes" / cat["path"]).mkdir(parents=True, exist_ok=True)


def _create_config(
    notes_path: Path,
    providers: list[str],
    categories: list[dict],
    threshold: float,
    routing_rules: list[dict],
    custom_providers: list[dict] = None
) -> None:
    """Create config.yaml."""
    custom_providers = custom_providers or []
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config = {
        "notes_path": str(notes_path),
        "folders": {
            "inbox": "_Inbox",
            "output": "Notes",
            "archive": "_Archive",
        },
        "llm": {
            "providers": {},
            "routing": routing_rules,
            "default_provider": routing_rules[-1]["provider"] if routing_rules else "claude",
        },
        "categories": categories,
        "fallback": {
            "enabled": True,
            "category": "_Unsorted",
            "review_flag": "[NEEDS REVIEW]",
        },
        "processing": {
            "confidence_threshold": threshold,
            "debounce_seconds": 2.0,
            "supported_extensions": [".txt", ".md"],
            "ignore_patterns": [".*", "_*", "~*"],
        },
        "log_level": "INFO",
    }

    # Create lookup dict for custom providers
    custom_lookup = {cp["id"]: cp for cp in custom_providers}

    # Add providers
    for provider in providers:
        # Check if it's a custom provider
        if provider in custom_lookup:
            custom = custom_lookup[provider]
            pconfig = {
                "model": custom["model"],
                "base_url": custom["base_url"],
                "enabled": True,
                "timeout": 60,
                "max_retries": 3,
            }
            if custom["env_key"]:
                pconfig["api_key_env"] = custom["env_key"]
        else:
            # Built-in provider
            info = PROVIDERS[provider]
            pconfig = {
                "model": info["default_model"],
                "enabled": True,
                "timeout": 60,
                "max_retries": 3,
            }
            if info["env_key"]:
                pconfig["api_key_env"] = info["env_key"]
            if "base_url" in info:
                pconfig["base_url"] = info["base_url"]

        config["llm"]["providers"][provider] = pconfig

    with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def _create_env_file(notes_path: Path, api_keys: dict[str, str], custom_providers: list[dict] = None) -> None:
    """Create .env file with API keys."""
    custom_providers = custom_providers or []
    custom_lookup = {cp["id"]: cp for cp in custom_providers}

    lines = ["# txttmd environment variables", ""]
    lines.append(f"NOTES_PATH={notes_path}")
    lines.append("")

    for provider, key in api_keys.items():
        env_key = None

        # Check if it's a custom provider
        if provider in custom_lookup:
            env_key = custom_lookup[provider]["env_key"]
        elif provider in PROVIDERS:
            env_key = PROVIDERS[provider]["env_key"]

        if env_key:
            lines.append(f"{env_key}={key}")

    with open(".env", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_wizard()
