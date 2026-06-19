"""Orchestrateur end-to-end (P0-6) — l'avion assemblé.

Déroule la chaîne complète pour un cas × des variables, sous traçage C15 :

    dossier (C04) → fiche (C05, gate R-1) → retrieval (C06)
    → qualification des preuves (C09) → contrôle temporel (C15)
    → suffisance (C07) ⇄ recherche active (C08, versée en C03, re-retrieval)
    → diagnostic (C09) → panel (C11/C10) → agrégation (C12)
    → réponse finale ou abstention (C13) → trace JSONL (C15)

CUSTOM (assumé) : pur câblage des composants existants — aucun algorithme.
Chaque étape est journalisée ; toute exception de garde-fou est FATALE pour
le cas (jamais avalée).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from compass.general_memory import GeneralMemory
from compass.country_memory import CountryMemory
from compass.party_election_case import CaseFileBuilder
from compass.vparty_registry import VPartyRegistry
from compass.internal_retrieval import InternalRetriever
from compass.sufficiency_gate import SufficiencyGate
from compass.active_search import ActiveSearchEngine
from compass.diagnostic_engine import DiagnosisEngine, EvidenceQualifier
from compass.reasoning_engine import ReasoningEngine
from compass.judge_panel import JudgePanel
from compass.aggregation import aggregate, aggregate_multilabel
from compass.final_output import AnswerComposer
from compass.guardrails import TraceLogger, assert_temporal_integrity
from compass.schemas import (CaseKey, FinalAnswer, OutputType, SufficiencyVerdict,
                     VariableMethod)

if TYPE_CHECKING:
    from compass.political_graph import PoliticalGraph

logger = logging.getLogger(__name__)


class CompassRunner:
    """Exécute la chaîne complète pour une unité pays × parti × élection."""

    def __init__(self, country: CountryMemory, general: GeneralMemory,
                 registry: VPartyRegistry, search: ActiveSearchEngine,
                 graph: "PoliticalGraph | None" = None) -> None:
        self._country = country
        self._registry = registry
        self._search = search
        self._builder = CaseFileBuilder(general, country, graph=graph)
        self._retriever = InternalRetriever(country=country)  # HyDE + parent enrichment
        self._qualifier = EvidenceQualifier()
        self._sufficiency = SufficiencyGate()
        self._diagnoser = DiagnosisEngine()
        self._panel = JudgePanel(ReasoningEngine(country))
        self._composer = AnswerComposer()

    def run_case(self, case: CaseKey, variable_ids: list[str]) -> list[FinalAnswer]:
        """Traite toutes les variables demandées pour un cas, avec traçabilité.

        Returns:
            Une FinalAnswer (score ou abstention) par variable lorsque le cas
            aboutit. Toute exception de garde-fou (gate R-1, fuite temporelle)
            interrompt le cas complet et remonte à l'appelant ; elle n'est
            jamais transformée silencieusement en abstention.
        """
        trace = TraceLogger(f"{case.country_iso3}_{case.party_id}_{case.election_id}")
        trace.record("case_start", case=case, variables=variable_ids)
        answers: list[FinalAnswer] = []

        dossier = self._builder.build(case)
        trace.record("dossier", n_party_docs=len(dossier.party_documents),
                     n_context=len(dossier.national_context))

        for var_id in variable_ids:
            sheet = self._registry.get(var_id, production=True)  # gate R-1
            trace.record("sheet", variable=var_id, method=sheet.method.value,
                         output_type=sheet.output_type.value)

            # --- boucle bornée : retrieval → qualification → suffisance → enquête
            searches_done = 0
            while True:
                passages = self._retriever.retrieve(dossier, sheet, case=case)
                evidence = self._qualifier.qualify(passages, sheet)
                assert_temporal_integrity(evidence, case.election_date)  # C15
                verdict, proba = self._sufficiency.decide(passages, sheet, searches_done)
                trace.record("sufficiency", variable=var_id,
                             verdict=verdict.value, proba=proba,
                             searches_done=searches_done)
                if verdict is not SufficiencyVerdict.SEARCH_MORE:
                    break
                diag_tmp = self._diagnoser.diagnose(case, sheet, evidence)
                new_segs = self._search.investigate(case, sheet, diag_tmp.missing)
                trace.record("active_search", variable=var_id,
                             n_new_segments=len(new_segs))
                if new_segs:
                    self._country.add_documents(new_segs)      # versés en C03
                    dossier = self._builder.build(case)        # dossier reconstruit
                searches_done += 1

            diagnosis = self._diagnoser.diagnose(case, sheet, evidence)
            diagnosis.graph_context = dossier.graph_context  # Gap 3
            trace.record("diagnosis", variable=var_id,
                         n_for=len(diagnosis.convergent),
                         n_against=len(diagnosis.contradictory),
                         contradictions=diagnosis.contradictions_detail,
                         missing=diagnosis.missing,
                         language=diagnosis.dominant_language)

            if verdict is SufficiencyVerdict.ABSTAIN:
                answer = self._composer.abstain(case, sheet, diagnosis, proba)
            else:
                judges = self._panel.evaluate(sheet, diagnosis)
                trace.record("panel", variable=var_id,
                             judges=[j.judge_id for j in judges])
                if sheet.output_type is OutputType.MULTI_LABEL:
                    judgment = aggregate_multilabel(judges)
                else:
                    judgment = aggregate(judges, method="median")
                answer = self._composer.compose(case, sheet, diagnosis,
                                                judgment, proba)
                # P0-3 : la sortie alimente compass_scores (entrée des dérivés)
                if answer.score is not None and sheet.method in (
                        VariableMethod.LLM_GUIDED, VariableMethod.NLP_CLASSIFIER,
                        VariableMethod.STRUCTURED_QUERY, VariableMethod.COMPOSITE):
                    self._country.store_compass_score(
                        case.party_id, case.election_id, var_id,
                        answer.score, answer.confidence or 0.0)

            answer.output_type = sheet.output_type
            trace.record("final", variable=var_id, abstained=answer.abstained,
                         score=answer.score, labels=answer.labels,
                         confidence=answer.confidence,
                         attribution_checked=answer.attribution_checked)
            answers.append(answer)

        trace.record("case_end", n_answers=len(answers),
                     n_abstentions=sum(a.abstained for a in answers))
        logger.info("Cas %s/%s : %d réponses (trace : %s)",
                    case.party_id, case.election_id, len(answers), trace.path)
        return answers
