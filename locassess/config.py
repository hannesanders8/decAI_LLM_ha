"""Configuration loading: config.yaml (research parameters) and .env (secrets)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocationCfg(_Strict):
    name: str
    region: str | None = None
    country: str
    local_language: str
    aliases: list[str] = Field(default_factory=list)


class PeriodCfg(_Strict):
    years_back: int = Field(5, ge=1)
    run_date: date | None = None


class CorpusCfg(_Strict):
    target_articles: int = Field(15, ge=1)
    max_per_year: int = Field(4, ge=1)
    min_words: int = Field(200, ge=0)
    queries_per_criterion: int = Field(2, ge=1)
    hits_per_query: int = Field(10, ge=1)
    min_relevance: int = Field(1, ge=0, le=2)


class ModelsCfg(_Strict):
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str
    temperature: float = 0
    decision_provider: Literal["jev", "openai_decisions", "llm_structured"] = "llm_structured"
    llm_confidence_method: Literal["logprobs", "sampling"] = "logprobs"
    sampling_n: int = Field(5, ge=1)
    max_options_per_question: int = Field(255, ge=2)


class RatingThresholds(_Strict):
    green: float = 1.0
    red: float = -1.0


class StatusCfg(_Strict):
    sufficient_min_evidence: int = 3
    sufficient_min_articles: int = 2


class ExtractionCfg(_Strict):
    affected_threshold: float = Field(0.5, ge=0, le=1)
    review_threshold: float = Field(0.6, ge=0, le=1)
    second_sentence_gap: float = Field(0.2, ge=0, le=1)


class RedFlagsCfg(_Strict):
    recent_months: int = 12
    knockout_criteria: list[str] = Field(default_factory=list)


class ScoringCfg(_Strict):
    type_weights: dict[Literal["fact", "announcement", "opinion"], float] = Field(
        default_factory=lambda: {"fact": 1.0, "announcement": 0.6, "opinion": 0.3}
    )
    half_life_months: float = Field(24, gt=0)
    rating_thresholds: RatingThresholds = Field(default_factory=RatingThresholds)
    trend_threshold: float = 1.0
    status: StatusCfg = Field(default_factory=StatusCfg)
    extraction: ExtractionCfg = Field(default_factory=ExtractionCfg)
    red_flags: RedFlagsCfg = Field(default_factory=RedFlagsCfg)


class ReportCfg(_Strict):
    max_attempts: int = Field(3, ge=1)


class AppConfig(_Strict):
    location: LocationCfg
    period: PeriodCfg = Field(default_factory=PeriodCfg)
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    whitelist_file: Path = Path("data/whitelist.csv")
    models: ModelsCfg
    search_provider: Literal["openai_web_search", "brave", "manual_csv"] = "openai_web_search"
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    weights_default: int = Field(1, ge=0, le=10)
    report: ReportCfg = Field(default_factory=ReportCfg)

    def resolved_run_date(self) -> date:
        """The configured run date, or today if none is fixed."""
        return self.period.run_date or date.today()


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)


def config_hash(cfg: AppConfig) -> str:
    """SHA-256 of the canonical JSON form of the configuration."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Secrets(BaseSettings):
    """API keys from .env. SecretStr keeps them out of repr() and logs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    typesafe_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    def values(self) -> list[str]:
        """All non-empty secret values (used to redact call logs)."""
        out = []
        for name in type(self).model_fields:
            v = getattr(self, name)
            if v is not None and v.get_secret_value():
                out.append(v.get_secret_value())
        return out
