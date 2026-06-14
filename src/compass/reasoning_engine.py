"""C10 — Moteur de raisonnement adaptable : la méthode dépend de la variable.

ÉTAT DE L'ART RÉUTILISÉ :
    - Patron de routage : Adaptive-RAG (Jeong et al. 2024) — adapter la
      stratégie au cas ; ici la clé de routage est ``sheet.method`` (régime
      de preuve), apport propre du projet.
    - STRUCTURED_QUERY : pandas/SQL sur la couche structurée C03 (V-Party,
      ParlGov importés) — zéro calcul maison.
    - DETERMINISTIC_RULE : formules officielles du codebook (ex. v2xpa_popul) —
      transcrites, pas inventées.
    - NLP_CLASSIFIER : transformers zero-shot NLI (Laurer et al. 2024,
      patron BERT-NLI déjà acté en Partie 3).
    - LLM_GUIDED : litellm — interface unifiée multi-fournisseurs (GPT/Mistral/
      Claude), température 0 (Ornstein et al. 2025).

CUSTOM (justifié) : le routeur lui-même (un dispatch ~20 lignes) et le prompt
guidé par fiche — le prompt EST la fiche C05 (définition, échelle, critères),
pas un texte libre : c'est la réponse à Halterman & Keith 2025.
"""

from __future__ import annotations

import json
import logging

import litellm
from transformers import pipeline as hf_pipeline

from compass.country_memory import CountryMemory
from compass.config import settings
from compass.schemas import Diagnosis, JudgeAnswer, VariableMethod, VariableSheet

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Route chaque variable vers sa méthode de traitement appropriée."""

    def __init__(self, country: CountryMemory) -> None:
        self._country = country
        self._zeroshot = hf_pipeline("zero-shot-classification", model=settings.nli_model)

    # ------------------------------------------------------------------ routage
    def answer(self, sheet: VariableSheet, diagnosis: Diagnosis,
               model_name: str, prompt_variant: str = "standard") -> JudgeAnswer:
        """Produit la réponse d'UN juge pour une variable d'un cas.

        Le couple (model_name, prompt_variant) identifie le juge — c'est C11
        qui orchestre la pluralité.
        """
        dispatch = {
            VariableMethod.STRUCTURED_QUERY: self._structured,
            VariableMethod.DETERMINISTIC_RULE: self._deterministic,
            VariableMethod.NLP_CLASSIFIER: self._classify,
            VariableMethod.LLM_GUIDED: self._llm_guided,
            VariableMethod.COMPOSITE: self._llm_guided,  # le LLM arbitre le composite
        }
        return dispatch[sheet.method](sheet, diagnosis, model_name, prompt_variant)

    # ------------------------------------------------------------------ méthodes
    def _structured(self, sheet: VariableSheet, diagnosis: Diagnosis,
                    model_name: str, _: str) -> JudgeAnswer:
        """Ex. v2pavote : lecture directe de la couche structurée — déterministe."""
        df = self._country.query_structured(
            "SELECT vote_share FROM results WHERE party_id = ? AND election_id = ?",
            (diagnosis.case.party_id, diagnosis.case.election_id),
        )
        score = float(df["vote_share"].iloc[0]) if not df.empty else float("nan")
        return JudgeAnswer(judge_id=f"structured::{sheet.variable_id}",
                           model_name="sqlite", score=score,
                           rationale="Lecture directe de la base structurée (import officiel).",
                           confidence=1.0 if not df.empty else 0.0)

    def _deterministic(self, sheet: VariableSheet, diagnosis: Diagnosis,
                       model_name: str, _: str) -> JudgeAnswer:
        """Ex. v2xpa_popul : formule du codebook sur variables élémentaires déjà codées."""
        rule = sheet.decision_rules[0] if sheet.decision_rules else ""
        # P0-3 : les formules dérivées consomment UNIQUEMENT les sorties
        # produites par COMPASS (compass_scores), jamais l'étalon V-Party.
        deps = self._country.query_structured(
            "SELECT variable_id, score FROM compass_scores WHERE party_id = ? AND election_id = ?",
            (diagnosis.case.party_id, diagnosis.case.election_id),
        )
        env = {row.variable_id: row.score for row in deps.itertuples()}
        try:
            score = float(eval(rule, {"__builtins__": {}}, env))  # noqa: S307 — formule du codebook, environnement clos
        except (NameError, SyntaxError, TypeError) as exc:
            logger.error("Formule inapplicable (%s) : %s", sheet.variable_id, exc)
            score, conf = float("nan"), 0.0
        else:
            conf = 1.0
        return JudgeAnswer(judge_id=f"rule::{sheet.variable_id}", model_name="formula",
                           score=score, rationale=f"Formule codebook : {rule}",
                           confidence=conf)

    def _classify(self, sheet: VariableSheet, diagnosis: Diagnosis,
                  model_name: str, _: str) -> JudgeAnswer:
        """Classification zero-shot guidée par les ancres de l'échelle.

        P1-6 — routage par langue : Political DEBATE (spécialisé textes
        politiques, Political Analysis 2025) pour l'anglais ; mDeBERTa-XNLI
        multilingue sinon. La langue vient du diagnostic (C09).
        """
        if diagnosis.dominant_language == "en":
            classifier = hf_pipeline("zero-shot-classification",
                                     model=settings.political_classifier)
            used = settings.political_classifier
        else:
            classifier = self._zeroshot
            used = settings.nli_model
        text = " ".join(e.segment.text for e in diagnosis.convergent[:8])
        labels = [f"{k}: {v}" for k, v in sheet.scale.items()]
        res = classifier(text or "aucune preuve", candidate_labels=labels)
        best = res["labels"][0]
        score = float(str(best).split(":")[0])
        return JudgeAnswer(judge_id=f"clf::{sheet.variable_id}::{used.split('/')[-1]}",
                           model_name=used, score=score,
                           rationale=f"Zero-shot ({used}), ancre retenue : {best}",
                           confidence=float(res["scores"][0]))

    def _llm_guided(self, sheet: VariableSheet, diagnosis: Diagnosis,
                    model_name: str, prompt_variant: str) -> JudgeAnswer:
        """Raisonnement multi-sources guidé par la fiche — le prompt EST la fiche."""
        prompt = self._build_prompt(sheet, diagnosis, prompt_variant)
        # Compatibilité litellm : response_format n'est pas supporté par tous
        # les fournisseurs — repli sans contrainte + parsing strict (audit §8).
        kwargs = dict(model=model_name, temperature=settings.llm_temperature,
                      max_tokens=settings.llm_max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        try:
            resp = litellm.completion(response_format={"type": "json_object"}, **kwargs)
        except litellm.BadRequestError:
            logger.warning("%s : response_format non supporté — repli texte.", model_name)
            resp = litellm.completion(**kwargs)
        payload = json.loads(resp.choices[0].message.content)
        return JudgeAnswer(
            judge_id=f"{model_name}::{prompt_variant}", model_name=model_name,
            score=float(payload["score"]), rationale=str(payload.get("rationale", "")),
            confidence=float(payload.get("confidence", 0.5)),
        )

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def _build_prompt(sheet: VariableSheet, diagnosis: Diagnosis,
                      variant: str) -> str:
        """Prompt = fiche C05 + diagnostic C09. Rien d'autre.

        Le juge reçoit définition, échelle ancrée, critères, preuves pour/contre
        et contradictions — et doit répondre en JSON avec régimes de preuve
        séparés (déclaré/observé/inféré).
        """
        ev_for = "\n".join(f"- [{e.regime.value}|{e.segment.meta.doc_type}] {e.segment.text}"
                           for e in diagnosis.convergent[:10])
        ev_against = "\n".join(f"- [{e.regime.value}|{e.segment.meta.doc_type}] {e.segment.text}"
                               for e in diagnosis.contradictory[:10])
        contradictions = "\n".join(f"- {c}" for c in diagnosis.contradictions_detail)
        scale = "\n".join(f"  {k} : {v}" for k, v in sheet.scale.items())
        style = ("Sois particulièrement attentif aux comportements observés, qui priment "
                 "sur les déclarations.") if variant == "behavior_first" else \
                ("Pèse déclarations et comportements à égalité, en signalant les écarts.")
        # Gap 3 — contexte relationnel du graphe de connaissances (C02b).
        graph_lines = "\n".join(
            f"- {g['summary']}" for g in (diagnosis.graph_context or [])[:6]
        )
        graph_section = (
            f"\nCONTEXTE RELATIONNEL [INFÉRÉ — cooccurrences documentaires] :\n"
            f"{graph_lines}\n"
            f"Ces relations sont des inférences statistiques, non des faits vérifiés. "
            f"Poids faible si elles contredisent les preuves documentaires ci-dessus.\n"
        ) if graph_lines else ""
        return f"""Tu codes la variable {sheet.variable_id} du codebook V-Party.
QUESTION : {sheet.question}
DÉFINITION (à appliquer strictement, mot à mot) : {sheet.definition}
ÉCHELLE :
{scale}
CRITÈRES D'INCLUSION : {'; '.join(sheet.inclusion_criteria)}
CRITÈRES D'EXCLUSION : {'; '.join(sheet.exclusion_criteria)}
{style}
{graph_section}
PREUVES EN FAVEUR :
{ev_for or '(aucune)'}
PREUVES CONTRAIRES :
{ev_against or '(aucune)'}
CONTRADICTIONS DÉTECTÉES :
{contradictions or '(aucune)'}

Réponds en JSON strict :
{{"score": <valeur de l'échelle>, "confidence": <0-1>, "rationale": "<justification citant les preuves par leur type>",
  "declared": ["..."], "observed": ["..."], "inferred": ["..."]}}
N'utilise AUCUNE connaissance extérieure aux preuves fournies. Si les preuves
sont insuffisantes, dis-le dans rationale et abaisse confidence."""

