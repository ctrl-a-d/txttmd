"""Configuration management for txttmd."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os
import yaml
from dotenv import load_dotenv


@dataclass
class FolderConfig:
    """Configuration for folder paths."""
    inbox: Path
    output: Path
    archive: Path

    @classmethod
    def from_dict(cls, data: dict, base_path: Path) -> "FolderConfig":
        """Create FolderConfig from dictionary."""
        return cls(
            inbox=base_path / data.get("inbox", "_Inbox"),
            output=base_path / data.get("output", "Notes"),
            archive=base_path / data.get("archive", "_Archive"),
        )


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""
    enabled: bool = True
    timeout: int = 60
    max_retries: int = 3

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ProviderConfig":
        """Create ProviderConfig from dictionary."""
        # Try to get API key from environment variable
        env_key = data.get("api_key_env")
        api_key = os.getenv(env_key) if env_key else data.get("api_key")

        return cls(
            name=name,
            api_key=api_key,
            base_url=data.get("base_url"),
            model=data.get("model", ""),
            enabled=data.get("enabled", True),
            timeout=data.get("timeout", 60),
            max_retries=data.get("max_retries", 3),
        )


@dataclass
class RoutingCondition:
    """A condition for routing to a specific provider."""
    condition_type: str  # word_count, contains_keywords, contains_code_blocks, always
    value: Any = None
    operator: str = "<"  # <, >, ==, <=, >=

    def evaluate(self, content: str) -> bool:
        """Evaluate the condition against content."""
        if self.condition_type == "always":
            return True

        if self.condition_type == "word_count":
            word_count = len(content.split())
            return self._compare(word_count, self.value)

        if self.condition_type == "contains_keywords":
            keywords = self.value if isinstance(self.value, list) else [self.value]
            return any(kw.lower() in content.lower() for kw in keywords)

        if self.condition_type == "contains_code_blocks":
            return "```" in content or "    " in content

        return False

    def _compare(self, actual: int, expected: int) -> bool:
        """Compare values based on operator."""
        ops = {
            "<": actual < expected,
            ">": actual > expected,
            "==": actual == expected,
            "<=": actual <= expected,
            ">=": actual >= expected,
        }
        return ops.get(self.operator, False)


@dataclass
class RoutingRule:
    """A routing rule mapping conditions to providers."""
    provider: str
    conditions: list[RoutingCondition] = field(default_factory=list)
    priority: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingRule":
        """Create RoutingRule from dictionary."""
        conditions = []
        for cond in data.get("conditions", []):
            conditions.append(RoutingCondition(
                condition_type=cond.get("type", "always"),
                value=cond.get("value"),
                operator=cond.get("operator", "<"),
            ))
        return cls(
            provider=data["provider"],
            conditions=conditions,
            priority=data.get("priority", 0),
        )


@dataclass
class LLMConfig:
    """Configuration for LLM providers and routing."""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routing_rules: list[RoutingRule] = field(default_factory=list)
    default_provider: str = "claude"

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """Create LLMConfig from dictionary."""
        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig.from_dict(name, pdata)

        routing_rules = []
        for rule in data.get("routing", []):
            routing_rules.append(RoutingRule.from_dict(rule))

        return cls(
            providers=providers,
            routing_rules=sorted(routing_rules, key=lambda r: -r.priority),
            default_provider=data.get("default_provider", "claude"),
        )


@dataclass
class Category:
    """Configuration for a note category."""
    name: str
    path: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Category":
        """Create Category from dictionary."""
        return cls(
            name=data["name"],
            path=data.get("path", data["name"]),
            keywords=data.get("keywords", []),
            description=data.get("description", ""),
        )


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enabled: bool = True
    category: str = "_Unsorted"
    review_flag: str = "[NEEDS REVIEW]"

    @classmethod
    def from_dict(cls, data: dict) -> "FallbackConfig":
        """Create FallbackConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            category=data.get("category", "_Unsorted"),
            review_flag=data.get("review_flag", "[NEEDS REVIEW]"),
        )


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior."""
    confidence_threshold: float = 0.7
    debounce_seconds: float = 2.0
    supported_extensions: list[str] = field(default_factory=lambda: [".txt", ".md"])
    ignore_patterns: list[str] = field(default_factory=lambda: [".*", "_*", "~*"])

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingConfig":
        """Create ProcessingConfig from dictionary."""
        return cls(
            confidence_threshold=data.get("confidence_threshold", 0.7),
            debounce_seconds=data.get("debounce_seconds", 2.0),
            supported_extensions=data.get("supported_extensions", [".txt", ".md"]),
            ignore_patterns=data.get("ignore_patterns", [".*", "_*", "~*"]),
        )


@dataclass
class Config:
    """Main configuration container."""
    folders: FolderConfig
    llm: LLMConfig
    categories: list[Category]
    fallback: FallbackConfig
    processing: ProcessingConfig
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def get_category_names(self) -> list[str]:
        """Get list of all category names."""
        return [cat.name for cat in self.categories]

    def get_category_by_name(self, name: str) -> Optional[Category]:
        """Get category by name (case-insensitive)."""
        for cat in self.categories:
            if cat.name.lower() == name.lower():
                return cat
        return None


class ConfigLoader:
    """Load and validate configuration from YAML and environment."""

    DEFAULT_CONFIG_PATH = Path("config/config.yaml")

    @classmethod
    def load(cls, config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> Config:
        """Load configuration from YAML file and environment variables."""
        # Load environment variables
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        # Determine config path
        if config_path is None:
            config_path = cls.DEFAULT_CONFIG_PATH

        config_path = Path(config_path)

        # Load YAML
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Determine base path for notes
        base_path_str = os.getenv("NOTES_PATH") or data.get("notes_path", ".")
        base_path = Path(base_path_str)

        # Build configuration
        folders = FolderConfig.from_dict(data.get("folders", {}), base_path)
        llm = LLMConfig.from_dict(data.get("llm", {}))

        categories = []
        for cat_data in data.get("categories", []):
            categories.append(Category.from_dict(cat_data))

        # Add default categories if none specified
        if not categories:
            categories = cls._default_categories()

        fallback = FallbackConfig.from_dict(data.get("fallback", {}))
        processing = ProcessingConfig.from_dict(data.get("processing", {}))

        log_file = data.get("log_file")
        if log_file:
            log_file = Path(log_file)

        return Config(
            folders=folders,
            llm=llm,
            categories=categories,
            fallback=fallback,
            processing=processing,
            log_level=os.getenv("LOG_LEVEL", data.get("log_level", "INFO")),
            log_file=log_file,
        )

    @classmethod
    def _default_categories(cls) -> list[Category]:
        """Return default categories."""
        return [
            Category(
                name="Projects",
                path="Projects",
                keywords=["project", "task", "todo", "milestone"],
                description="Project-related notes",
            ),
            Category(
                name="Ideas",
                path="Ideas",
                keywords=["idea", "brainstorm", "concept", "thought"],
                description="Ideas and brainstorming",
            ),
            Category(
                name="Research",
                path="Research",
                keywords=["research", "study", "paper", "article"],
                description="Research notes",
            ),
            Category(
                name="Personal",
                path="Personal",
                keywords=["personal", "journal", "diary", "reflection"],
                description="Personal notes",
            ),
            Category(
                name="Work",
                path="Work",
                keywords=["work", "meeting", "client", "business"],
                description="Work-related notes",
            ),
        ]

    @classmethod
    def save(cls, config: Config, config_path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "notes_path": str(config.folders.inbox.parent),
            "folders": {
                "inbox": config.folders.inbox.name,
                "output": config.folders.output.name,
                "archive": config.folders.archive.name,
            },
            "llm": {
                "providers": {},
                "routing": [],
                "default_provider": config.llm.default_provider,
            },
            "categories": [],
            "fallback": {
                "enabled": config.fallback.enabled,
                "category": config.fallback.category,
                "review_flag": config.fallback.review_flag,
            },
            "processing": {
                "confidence_threshold": config.processing.confidence_threshold,
                "debounce_seconds": config.processing.debounce_seconds,
                "supported_extensions": config.processing.supported_extensions,
                "ignore_patterns": config.processing.ignore_patterns,
            },
            "log_level": config.log_level,
        }

        # Add providers
        for name, provider in config.llm.providers.items():
            pdata = {
                "model": provider.model,
                "enabled": provider.enabled,
                "timeout": provider.timeout,
                "max_retries": provider.max_retries,
            }
            if provider.base_url:
                pdata["base_url"] = provider.base_url
            # Store env var reference instead of actual key
            pdata["api_key_env"] = f"{name.upper()}_API_KEY"
            data["llm"]["providers"][name] = pdata

        # Add routing rules
        for rule in config.llm.routing_rules:
            rdata = {
                "provider": rule.provider,
                "priority": rule.priority,
                "conditions": [],
            }
            for cond in rule.conditions:
                cdata = {"type": cond.condition_type}
                if cond.value is not None:
                    cdata["value"] = cond.value
                if cond.operator != "<":
                    cdata["operator"] = cond.operator
                rdata["conditions"].append(cdata)
            data["llm"]["routing"].append(rdata)

        # Add categories
        for cat in config.categories:
            data["categories"].append({
                "name": cat.name,
                "path": cat.path,
                "keywords": cat.keywords,
                "description": cat.description,
            })

        if config.log_file:
            data["log_file"] = str(config.log_file)

        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
