"""LLM providers and note processing for txttmd."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

from .config import LLMConfig, ProviderConfig, RoutingRule

logger = logging.getLogger(__name__)


# Prompt templates
TXT_PROMPT_TEMPLATE = """You are a note organization assistant. Analyze the following raw note and transform it into a well-structured markdown document.

## Available Categories
{categories}

## Instructions
1. Analyze the content and determine the most appropriate category
2. Generate a descriptive filename (without extension)
3. Transform the content into clean, well-formatted markdown
4. Extract relevant tags from the content
5. List any significant changes you made to the content

## Input Note
```
{content}
```

## Response Format
Respond ONLY with valid JSON in this exact format:
{{
    "category": "chosen category name",
    "confidence": 0.85,
    "reasoning": "brief explanation of why this category was chosen",
    "filename": "descriptive-filename",
    "content": "# Title\\n\\nTransformed markdown content...",
    "tags": ["tag1", "tag2"],
    "changes_made": ["Added title", "Formatted as list", "Fixed typos"]
}}"""

MD_PROMPT_TEMPLATE = """You are a note organization assistant. Analyze the following markdown note and enhance it if needed.

## Available Categories
{categories}

## Instructions
1. Analyze the content and determine the most appropriate category
2. Generate a descriptive filename (without extension) based on content
3. Enhance the markdown structure if needed (add headings, fix formatting)
4. Extract relevant tags from the content
5. List any changes made (or empty list if no changes)

## Input Note
```markdown
{content}
```

## Response Format
Respond ONLY with valid JSON in this exact format:
{{
    "category": "chosen category name",
    "confidence": 0.85,
    "reasoning": "brief explanation of why this category was chosen",
    "filename": "descriptive-filename",
    "content": "# Title\\n\\nEnhanced markdown content...",
    "tags": ["tag1", "tag2"],
    "changes_made": ["Added heading", "Improved structure"]
}}"""


@dataclass
class NoteResult:
    """Result from processing a note through an LLM."""
    category: str
    confidence: float
    reasoning: str
    filename: str
    content: str
    tags: list[str] = field(default_factory=list)
    changes_made: Optional[list[str]] = None
    provider_used: str = ""

    @classmethod
    def from_json(cls, data: dict, provider: str = "") -> "NoteResult":
        """Create NoteResult from JSON response."""
        return cls(
            category=data.get("category", ""),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            filename=data.get("filename", "untitled"),
            content=data.get("content", ""),
            tags=data.get("tags", []),
            changes_made=data.get("changes_made"),
            provider_used=provider,
        )


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        """Initialize provider with configuration."""
        self.config = config
        self.name = config.name

    @abstractmethod
    def process_note(
        self,
        content: str,
        file_type: str,
        categories: list[str],
    ) -> NoteResult:
        """
        Process a note and return structured result.

        Args:
            content: Raw note content.
            file_type: File extension (e.g., ".txt", ".md").
            categories: List of available category names.

        Returns:
            NoteResult with categorization and transformed content.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is available and working.

        Returns:
            True if provider is healthy.
        """
        pass

    def _get_prompt(self, content: str, file_type: str, categories: list[str]) -> str:
        """Get appropriate prompt template based on file type."""
        categories_str = "\n".join(f"- {cat}" for cat in categories)

        if file_type.lower() == ".md":
            template = MD_PROMPT_TEMPLATE
        else:
            template = TXT_PROMPT_TEMPLATE

        return template.format(content=content, categories=categories_str)

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (code block markers)
            lines = [l for l in lines if not l.startswith("```")]
            response = "\n".join(lines)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
            raise


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def process_note(
        self,
        content: str,
        file_type: str,
        categories: list[str],
    ) -> NoteResult:
        """Process note using Claude."""
        prompt = self._get_prompt(content, file_type, categories)

        response = self.client.messages.create(
            model=self.config.model or "claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        data = self._parse_response(response_text)
        return NoteResult.from_json(data, self.name)

    def health_check(self) -> bool:
        """Check Claude API availability."""
        try:
            self.client.messages.create(
                model=self.config.model or "claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            return False


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible provider (works with OpenAI, Groq, Gemini, Mistral, OpenRouter)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def process_note(
        self,
        content: str,
        file_type: str,
        categories: list[str],
    ) -> NoteResult:
        """Process note using OpenAI-compatible API."""
        prompt = self._get_prompt(content, file_type, categories)

        response = self.client.chat.completions.create(
            model=self.config.model or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )

        response_text = response.choices[0].message.content
        data = self._parse_response(response_text)
        return NoteResult.from_json(data, self.name)

    def health_check(self) -> bool:
        """Check API availability."""
        try:
            self.client.chat.completions.create(
                model=self.config.model or "gpt-4o",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            logger.warning(f"{self.name} health check failed: {e}")
            return False


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise ImportError("ollama package not installed. Run: pip install ollama")
        return self._client

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def process_note(
        self,
        content: str,
        file_type: str,
        categories: list[str],
    ) -> NoteResult:
        """Process note using Ollama."""
        prompt = self._get_prompt(content, file_type, categories)

        response = self.client.chat(
            model=self.config.model or "llama3.2",
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response["message"]["content"]
        data = self._parse_response(response_text)
        return NoteResult.from_json(data, self.name)

    def health_check(self) -> bool:
        """Check Ollama availability."""
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False


class PerplexityProvider(LLMProvider):
    """Perplexity API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy-load httpx client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.config.base_url or "https://api.perplexity.ai",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise ImportError("httpx package not installed. Run: pip install httpx")
        return self._client

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def process_note(
        self,
        content: str,
        file_type: str,
        categories: list[str],
    ) -> NoteResult:
        """Process note using Perplexity."""
        prompt = self._get_prompt(content, file_type, categories)

        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.config.model or "llama-3.1-sonar-large-128k-online",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()

        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
        parsed = self._parse_response(response_text)
        return NoteResult.from_json(parsed, self.name)

    def health_check(self) -> bool:
        """Check Perplexity API availability."""
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.config.model or "llama-3.1-sonar-large-128k-online",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Perplexity health check failed: {e}")
            return False


class ProviderFactory:
    """Factory for creating LLM providers."""

    PROVIDERS = {
        "claude": ClaudeProvider,
        "anthropic": ClaudeProvider,
        "openai": OpenAICompatibleProvider,
        "groq": OpenAICompatibleProvider,
        "gemini": OpenAICompatibleProvider,
        "mistral": OpenAICompatibleProvider,
        "openrouter": OpenAICompatibleProvider,
        "ollama": OllamaProvider,
        "perplexity": PerplexityProvider,
    }

    @classmethod
    def create(cls, provider_name: str, config: ProviderConfig) -> LLMProvider:
        """
        Create a provider instance.

        Args:
            provider_name: Name of the provider (e.g., "claude", "openai").
            config: Provider configuration.

        Returns:
            LLMProvider instance.

        Raises:
            ValueError: If provider is not supported.
        """
        provider_key = provider_name.lower()

        if provider_key not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Supported: {list(cls.PROVIDERS.keys())}"
            )

        return cls.PROVIDERS[provider_key](config)


class SmartRouter:
    """Route notes to appropriate providers based on content."""

    def __init__(self, rules: list[RoutingRule], default_provider: str):
        """
        Initialize router.

        Args:
            rules: List of routing rules (already sorted by priority).
            default_provider: Default provider if no rules match.
        """
        self.rules = rules
        self.default_provider = default_provider

    def get_provider(self, content: str) -> str:
        """
        Determine which provider to use for content.

        Args:
            content: Note content.

        Returns:
            Provider name.
        """
        for rule in self.rules:
            if all(cond.evaluate(content) for cond in rule.conditions):
                logger.debug(f"Routing to {rule.provider} based on rule match")
                return rule.provider

        logger.debug(f"Using default provider: {self.default_provider}")
        return self.default_provider


class NoteProcessor:
    """Main note processing orchestrator."""

    def __init__(self, llm_config: LLMConfig, categories: list[str]):
        """
        Initialize processor.

        Args:
            llm_config: LLM configuration with providers and routing.
            categories: List of available category names.
        """
        self.categories = categories
        self.providers: dict[str, LLMProvider] = {}
        self.router = SmartRouter(llm_config.routing_rules, llm_config.default_provider)
        self.default_provider = llm_config.default_provider

        # Initialize enabled providers
        for name, config in llm_config.providers.items():
            if config.enabled:
                try:
                    self.providers[name] = ProviderFactory.create(name, config)
                    logger.info(f"Initialized provider: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize provider {name}: {e}")

        if not self.providers:
            raise ValueError("No LLM providers could be initialized")

    def process(self, content: str, file_type: str) -> NoteResult:
        """
        Process a note through the appropriate provider.

        Args:
            content: Note content.
            file_type: File extension.

        Returns:
            NoteResult with categorization and content.

        Raises:
            RuntimeError: If all providers fail.
        """
        # Determine primary provider
        primary = self.router.get_provider(content)

        # Build fallback chain
        providers_to_try = []
        if primary in self.providers:
            providers_to_try.append(primary)

        # Add other providers as fallbacks
        for name in self.providers:
            if name not in providers_to_try:
                providers_to_try.append(name)

        # Try each provider
        last_error = None
        for provider_name in providers_to_try:
            try:
                logger.info(f"Processing with provider: {provider_name}")
                provider = self.providers[provider_name]
                result = provider.process_note(content, file_type, self.categories)
                result.provider_used = provider_name
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def health_check_all(self) -> dict[str, bool]:
        """
        Check health of all providers.

        Returns:
            Dictionary mapping provider name to health status.
        """
        results = {}
        for name, provider in self.providers.items():
            results[name] = provider.health_check()
        return results
