"""Scientific COMPASS pipeline exposed through the conversational facade."""

from __future__ import annotations

from datetime import date
from typing import Any

from compass.schemas import CaseKey, FinalAnswer


class ScientificConfigurationError(RuntimeError):
    """The active chat session lacks data required by the scientific pipeline."""


class ScientificChatService:
    """Lazy adapter from chat commands to the existing C04-C15 pipeline."""

    def __init__(
        self,
        memory: Any,
        graph: Any | None = None,
        *,
        runner: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        self.memory = memory
        self.graph = graph
        self._runner = runner
        self._registry = registry
        self._answers: list[FinalAnswer] = []

    def analyze(
        self,
        variable_id: str,
        *,
        country_iso3: str,
        party_id: str | None,
        election_id: str | None,
        as_of: date,
    ) -> FinalAnswer:
        """Run one registered scientific variable for the active party-election case."""
        resolved_party = party_id or self._single_indexed_party_id()
        if not resolved_party:
            raise ScientificConfigurationError(
                "L'analyse scientifique exige un party_id unique dans la session."
            )
        resolved_election = election_id or self._resolve_election_id(country_iso3, as_of)
        if not resolved_election:
            raise ScientificConfigurationError(
                "Aucun election_id n'est disponible. Passez --election-id ou importez la table elections."
            )
        registry = self._ensure_registry()
        if variable_id not in registry.list_ids():
            available = ", ".join(registry.list_ids()) or "aucune"
            raise ScientificConfigurationError(
                f"Variable inconnue : {variable_id}. Variables disponibles : {available}."
            )
        registry.get(variable_id, production=True)
        runner, registry = self._ensure_runtime()
        case = CaseKey(
            country_iso3=country_iso3.upper(),
            party_id=resolved_party,
            election_id=resolved_election,
            election_date=as_of,
        )
        answers = runner.run_case(case, [variable_id])
        if not answers:
            raise RuntimeError("Le pipeline scientifique n'a produit aucune réponse.")
        answer = answers[0]
        self._answers.append(answer)
        return answer

    def validate_cached(self, variable_id: str | None = None) -> Any:
        """Evaluate produced answers against the physically separate C14 vault."""
        selected = [
            answer for answer in self._answers
            if variable_id is None or answer.variable_id == variable_id
        ]
        if not selected:
            raise ScientificConfigurationError(
                "Aucune analyse scientifique de cette session n'est disponible à valider."
            )
        from compass.validation import EvaluationVault, Validator

        truth = EvaluationVault().truth()
        if truth.empty:
            raise ScientificConfigurationError(
                "Le coffre C14 ne contient aucun score étalon. Importez V-Party avant la validation."
            )
        return Validator(truth).evaluate(selected, stratum="chat_session")

    def contamination_check(
        self,
        variable_id: str,
        *,
        party_id: str | None,
        election_year: int,
    ) -> list[dict]:
        """Run the explicit C15 memorization probe outside production reasoning."""
        resolved_party = party_id or self._single_indexed_party_id()
        if not resolved_party:
            raise ScientificConfigurationError(
                "La sonde de contamination exige un party_id unique."
            )
        party_name = resolved_party
        try:
            frame = self.memory.query_structured(
                "SELECT name FROM parties WHERE party_id = ? LIMIT 1",
                (resolved_party,),
            )
            if not frame.empty:
                party_name = str(frame.iloc[0]["name"] or resolved_party)
        except Exception:
            pass

        from compass.config import settings
        from compass.guardrails import contamination_probe

        return [
            contamination_probe(model, party_name, election_year, variable_id)
            for model in settings.judge_models
        ]

    def available_variables(self) -> list[str]:
        registry = self._ensure_registry()
        if not hasattr(registry, "get"):
            return registry.list_ids()
        ready = []
        for variable_id in registry.list_ids():
            try:
                registry.get(variable_id, production=True)
            except Exception:
                continue
            ready.append(variable_id)
        return ready

    @property
    def last_trace_path(self) -> str | None:
        path = getattr(self._runner, "last_trace_path", None)
        return str(path) if path else None

    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._runner is not None and self._registry is not None:
            return self._runner, self._registry

        from compass.active_search import ActiveSearchEngine
        from compass.document_pipeline import DocumentPipeline
        from compass.general_memory import GeneralMemory
        from compass.orchestrator import CompassRunner
        self._registry = self._ensure_registry()
        self._runner = CompassRunner(
            country=self.memory,
            general=GeneralMemory(),
            registry=self._registry,
            search=ActiveSearchEngine(DocumentPipeline()),
            graph=self.graph,
        )
        return self._runner, self._registry

    def _ensure_registry(self) -> Any:
        if self._registry is None:
            from compass.vparty_registry import VPartyRegistry

            self._registry = VPartyRegistry()
        return self._registry

    def _single_indexed_party_id(self) -> str | None:
        try:
            scope = self.memory.describe_corpus()
        except Exception:
            return None
        parties = scope.get("parties") or []
        if len(parties) != 1 or not isinstance(parties[0], dict):
            return None
        value = str(parties[0].get("party_id") or "").strip()
        return value or None

    def _resolve_election_id(self, country_iso3: str, as_of: date) -> str | None:
        try:
            frame = self.memory.query_structured(
                "SELECT election_id FROM elections "
                "WHERE country_iso3 = ? AND election_date <= ? "
                "ORDER BY election_date DESC LIMIT 1",
                (country_iso3.upper(), as_of.isoformat()),
            )
        except Exception:
            return None
        if frame.empty:
            return None
        value = str(frame.iloc[0]["election_id"] or "").strip()
        return value or None
