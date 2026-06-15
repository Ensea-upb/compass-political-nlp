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

import ast
import json
import logging
import operator

from compass.country_memory import CountryMemory
from compass.config import settings
from compass.llm_client import complete_chat
from compass.nlp_models import political_classifier_pipeline, zero_shot_pipeline
from compass.schemas import Diagnosis, JudgeAnswer, VariableMethod, VariableSheet

_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

logger = logging.getLogger(__name__)


def safe_eval_formula(rule: str, env: dict[str, float]) -> float:
    """Evaluate a numeric codebook formula without Python dynamic evaluation.

    Allowed grammar: numeric constants, variable names present in ``env``,
    +, -, *, /, parentheses, and unary +/-.
    """
    if not rule or not rule.strip():
        raise ValueError("formule vide")
    tree = ast.parse(rule, mode="eval")
    return float(_eval_formula_node(tree.body, env))


def _eval_formula_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NameError(f"variable inconnue: {node.id}")
        return float(env[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        left = _eval_formula_node(node.left, env)
        right = _eval_formula_node(node.right, env)
        return float(_SAFE_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return float(_SAFE_UNARY_OPS[type(node.op)](_eval_formula_node(node.operand, env)))
    raise ValueError(f"expression non autorisée: {type(node).__name__}")


def parse_scale_score(label: object) -> float:
    """Parse a numeric scale prefix such as '3: committed'."""
    prefix = str(label).split(":", 1)[0].strip()
    return float(prefix)

class ReasoningEngine:
    """Route chaque variable vers sa méthode de traitement appropriée."""

    def __init__(self, country: CountryMemory) -> None:
        self._country = country
        self._zeroshot = zero_shot_pipeline()

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
            score = float(safe_eval_formula(rule, env))
        except (NameError, SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
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
            classifier = political_classifier_pipeline()
            used = settings.political_classifier
        else:
            classifier = self._zeroshot
            used = settings.nli_model
        text = " ".join(e.segment.text for e in diagnosis.convergent[:8])
        labels = [f"{k}: {v}" for k, v in sheet.scale.items()]
        res = classifier(text or "aucune preuve", candidate_labels=labels)
        best = res["labels"][0]
        try:
            score = parse_scale_score(best)
            confidence = float(res["scores"][0])
        except (TypeError, ValueError) as exc:
            logger.error("Label non numérique (%s) : %s", sheet.variable_id, exc)
            score, confidence = float("nan"), 0.0
        return JudgeAnswer(judge_id=f"clf::{sheet.variable_id}::{used.split('/')[-1]}",
                           model_name=used, score=score,
                           rationale=f"Zero-shot ({used}), ancre retenue : {best}",
                           confidence=confidence)

    def _llm_guided(self, sheet: VariableSheet, diagnosis: Diagnosis,
                    model_name: str, prompt_variant: str) -> JudgeAnswer:
        """Raisonnement multi-sources guidé par la fiche — le prompt EST la fiche."""
        prompt = self._build_prompt(sheet, diagnosis, prompt_variant)
        raw = complete_chat(
            model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
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
