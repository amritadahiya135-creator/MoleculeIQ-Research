"""
Lightweight Literature Polarity and Claim Directionality Annotation Layer (Module M09-D).

Scientific Safeguards & Epistemological Boundaries:
---------------------------------------------------
1. Deterministic Rule-Based Linguistic Annotation:
   Labels describe explicit textual wording in retrieved biomedical sentences.
   They do NOT represent biological truth, clinical efficacy, causality, or experimental reproducibility.
2. Conservative Classification Philosophy:
   - AFFIRMATIVE: The sentence contains wording asserting a reported association,
     causal mechanism, or adverse effect between the queried entity and biological endpoint,
     in the absence of negation or uncertainty.
   - NEGATED: The sentence contains explicit linguistic refutation or denial of the
     association or adverse outcome (e.g., 'did not induce', 'no evidence of',
     'failed to cause', 'without detectable toxicity').
     IMPORTANT: Negation in an abstract does NOT prove compound safety or absence of hazard.
   - UNCERTAIN: Default label for hedging ('may cause', 'might induce', 'suggests'),
     mixed or conflicting clauses ('did not cause at low doses but induced at high doses'),
     statistical equivalence ('no significant difference'), complex cross-clause scope,
     or ambiguous phrasing.
3. False-Positive & Scope Guards:
   - Non-negating additive idioms ('not only ... but also') are protected from false negation.
   - Comparative statistical statements ('no significant difference') are flagged as UNCERTAIN.
   - Word boundaries prevent collisions on substrings ('not' inside 'notable', 'noticeable', 'knot').
   - Multi-clause contrastive conjunctions ('but', 'however', 'whereas') with opposing cues trigger UNCERTAIN.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, List
import re

POLARITY_AFFIRMATIVE = "AFFIRMATIVE"
POLARITY_NEGATED = "NEGATED"
POLARITY_UNCERTAIN = "UNCERTAIN"

POLARITY_LABELS: Set[str] = {
    POLARITY_AFFIRMATIVE,
    POLARITY_NEGATED,
    POLARITY_UNCERTAIN,
}

POLARITY_UI_LABELS: Dict[str, str] = {
    POLARITY_AFFIRMATIVE: "Possible affirmative claim",
    POLARITY_NEGATED: "Negated wording",
    POLARITY_UNCERTAIN: "Polarity uncertain",
}

POLARITY_EXPLANATIONS: Dict[str, str] = {
    POLARITY_AFFIRMATIVE: (
        "Wording appears to assert the reported relationship in the retrieved text. "
        "Heuristic text annotation only; does not confirm biological causality or experimental reproducibility."
    ),
    POLARITY_NEGATED: (
        "Wording explicitly negates or refutes the reported relationship in the retrieved text. "
        "Heuristic text annotation only; does not prove compound safety or absence of hazard under other conditions."
    ),
    POLARITY_UNCERTAIN: (
        "Wording exhibits hedging, mixed evidence, conflicting clauses, or indeterminate polarity. "
        "Heuristic text annotation only; polarity cannot be deterministically inferred from simple patterns."
    ),
}


@dataclass(frozen=True)
class PolarityAnnotation:
    """Structured linguistic polarity assessment with full provenance and explanation."""
    label: str
    ui_label: str
    explanation: str
    matched_cue: Optional[str] = None
    confidence_rationale: Optional[str] = None


# Patterns for additive correlative conjunctions that contain 'not' without negating the predicate
RE_ADDITIVE_NOT_ONLY = re.compile(
    r"\bnot\s+(?:only|just)\b",
    re.IGNORECASE,
)

# Patterns for statistical comparative equivalence (ambiguous; not proof of absence of toxicity)
RE_NO_SIGNIFICANT_DIFF = re.compile(
    r"\bno\s+(?:statistically\s+)?(?:significant|marked)\s+(?:difference|change|alteration|effect|increase|decrease|correlation|association)\b",
    re.IGNORECASE,
)

# Contrastive conjunctions indicating potential clause shift or mixed findings
RE_CONTRASTIVE_CONJUNCTION = re.compile(
    r"\b(?:but|however|whereas|although|yet|nevertheless|conversely|in\s+contrast|on\s+the\s+other\s+hand)\b",
    re.IGNORECASE,
)

# Dose-dependent contrast indicators (e.g. "at low doses ... but at high doses")
RE_DOSE_CONTRAST = re.compile(
    r"\b(?:at|in)\s+(?:low|lower|sub-toxic|therapeutic)\s+(?:doses?|concentrations?).+?\b(?:at|in)\s+(?:high|higher|supratherapeutic|elevated)\s+(?:doses?|concentrations?)\b",
    re.IGNORECASE,
)

# Explicit Negation Patterns targeting relationship refutation or adverse event denial
NEGATION_PATTERNS = [
    # Negated action verbs: "did not induce", "does not cause", "was not associated with"
    re.compile(
        r"\b(?:did\s+not|does\s+not|do\s+not|was\s+not|were\s+not|is\s+not|are\s+not|has\s+not|have\s+not|had\s+not)\s+"
        r"(?:significantly\s+|substantially\s+|directly\s+)?"
        r"(?:induce|cause|produce|elicit|trigger|precipitate|provoke|promote|lead\s+to|result\s+in|aggravate|exacerbate|worsen|elevate|increase)\b",
        re.IGNORECASE,
    ),
    # "was not associated with", "were not linked to"
    re.compile(
        r"\b(?:was\s+not|were\s+not|is\s+not|are\s+not)\s+"
        r"(?:significantly\s+|consistently\s+)?"
        r"(?:associated\s+with|correlated\s+with|linked\s+to|causally\s+related\s+to)\b",
        re.IGNORECASE,
    ),
    # "no evidence of"
    re.compile(
        r"\bno\s+evidence\s+of\b",
        re.IGNORECASE,
    ),
    # "failed to cause / induce / produce"
    re.compile(
        r"\b(?:failed|fails|failing)\s+to\s+(?:cause|induce|produce|elicit|trigger|show|demonstrate|precipitate|elevate)\b",
        re.IGNORECASE,
    ),
    # "without detectable / observable / apparent / significant toxicity / injury"
    re.compile(
        r"\bwithout\s+(?:any\s+|detectable\s+|apparent\s+|demonstrable\s+|significant\s+|observable\s+|obvious\s+)?"
        r"(?:toxicity|adverse\s+effects?|toxic\s+effects?|hepatotoxicity|cardiotoxicity|nephrotoxicity|liver\s+injury|organ\s+damage|damage|lethality|mortality|acidosis)\b",
        re.IGNORECASE,
    ),
    # "absence of toxicity / damage / adverse effects"
    re.compile(
        r"\babsence\s+of\s+(?:detectable\s+|observable\s+|apparent\s+|significant\s+)?"
        r"(?:toxicity|adverse\s+effects?|toxic\s+effects?|liver\s+injury|damage|necrosis|acidosis|cardiotoxicity|hepatotoxicity)\b",
        re.IGNORECASE,
    ),
    # "no toxicity was observed / detected / found"
    re.compile(
        r"\bno\s+(?:detectable\s+|observable\s+|apparent\s+|significant\s+)?"
        r"(?:toxicity|adverse\s+effects?|toxic\s+effects?|hepatotoxicity|cardiotoxicity|nephrotoxicity|histopathological\s+changes?|liver\s+damage)\s+"
        r"(?:was|were|could\s+be)\s+(?:observed|detected|noted|found|seen|demonstrated)\b",
        re.IGNORECASE,
    ),
]

# Explicit Hedging and Epistemological Uncertainty Patterns
UNCERTAINTY_PATTERNS = [
    # Modal auxiliaries: "may cause", "might induce", "could be associated with"
    re.compile(
        r"\b(?:may|might|could)\s+(?:potentially\s+)?"
        r"(?:cause|induce|lead\s+to|result\s+in|precipitate|trigger|promote|aggravate|exacerbate|worsen|elevate|increase)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:may|might|could)\s+be\s+(?:associated\s+with|linked\s+to|causally\s+related\s+to|responsible\s+for)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmay\s+have\s+(?:caused|induced|contributed\s+to|precipitated)\b",
        re.IGNORECASE,
    ),
    # Adverbs and epistemic hedging phrases
    re.compile(
        r"\bpossibly\s+(?:linked|associated|caused|induced|related|due\s+to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpotentially\s+(?:toxic|harmful|damaging|lethal|hazardous|fatal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpotential\s+(?:risk|toxicity|adverse\s+effects?|association|link)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:suggests?|suggesting|appears?\s+to|seems?\s+to)\s+(?:indicate|suggest|cause|induce|increase|be\s+associated|elevate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:suggests?|suggesting|indicated?|indicating)\s+that\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:purported(?:ly)?|hypothesized\s+to|plausible\s+mechanism)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:remains?\s+(?:unclear|controversial|inconclusive|unproven|debated|unknown))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot\s+be\s+ruled\s+out|cannot\s+be\s+excluded|can\s+not\s+be\s+ruled\s+out)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:further\s+(?:studies|investigations?|research)\s+(?:are|is)\s+(?:needed|warranted|required))\b",
        re.IGNORECASE,
    ),
]

# Reusable Adverse Endpoint Terms pattern
ADVERSE_ENDPOINT_TERMS = (
    r"(?:(?:severe|acute|marked|significant|dose-dependent|fatal|pronounced|profound|toxic)\s+)*"
    r"(?:toxicity|hepatotoxicity|cardiotoxicity|nephrotoxicity|neurotoxicity|cytotoxicity|"
    r"liver\s+(?:injury|damage|failure|necrosis)|hepatic\s+(?:damage|injury|failure|necrosis)|"
    r"renal\s+(?:injury|damage|failure)|cardiac\s+(?:injury|damage|arrest|failure|toxicity)|"
    r"organ\s+(?:damage|injury|failure)|tissue\s+(?:injury|damage|necrosis)|"
    r"necrosis|damage|acidosis|lactic\s+acidosis|hyperlactatemia|mala|"
    r"apoptosis|failure|death|lethality|mortality|circulatory\s+collapse)"
)

# Explicit Affirmative Causation & Association Patterns
AFFIRMATIVE_PATTERNS = [
    # Active causation verbs linking agent to toxic/adverse outcome
    re.compile(
        rf"\b(?:induced|induces|inducing)\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:caused|causes|causing)\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:resulted\s+in|results\s+in|led\s+to|leads\s+to)\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:precipitated|precipitates|precipitating|triggered|triggers|triggering|elicited|elicits|eliciting)\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    # Documented clinical/experimental association or elevation
    re.compile(
        rf"\b(?:was|were|is|are)\s+(?:significantly\s+|strongly\s+|consistently\s+)?"
        rf"associated\s+with\s+(?:increased\s+|elevated\s+|higher\s+|excess\s+)?"
        rf"(?:risk\s+of\s+|incidence\s+of\s+|rates?\s+of\s+)?{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:significantly\s+|markedly\s+|dose-dependently\s+)"
        r"(?:increased|elevated|aggravated|exacerbated|worsened)\s+"
        r"(?:the\s+risk\s+of\s+|serum\s+transaminases|transaminase\s+levels|lactate\s+levels|liver\s+injury|liver\s+damage|toxicity|cardiotoxicity|mortality)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bproduced\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bdemonstrated\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bprimary\s+cause\s+of\s+{ADVERSE_ENDPOINT_TERMS}\b",
        re.IGNORECASE,
    ),
]


def classify_claim_polarity(
    sentence: str,
    query: Optional[str] = None,
) -> PolarityAnnotation:
    """
    Deterministically classify the linguistic polarity of a biomedical sentence.

    Parameters
    ----------
    sentence : str
        The raw extracted biomedical sentence.
    query : Optional[str], default=None
        The search query context (optional, reserved for future query-grounded scoring).

    Returns
    -------
    PolarityAnnotation
        Immutable structured annotation with label ('AFFIRMATIVE', 'NEGATED', 'UNCERTAIN'),
        UI display badge text, and detailed epistemological explanation.
    """
    # Guard against empty, whitespace-only, or non-string inputs
    if not sentence or not isinstance(sentence, str) or not sentence.strip():
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation="Empty or missing sentence input.",
            confidence_rationale="No textual tokens available for linguistic evaluation.",
        )

    clean_text = sentence.strip()

    # Guard against punctuation-only or symbol-only inputs
    if not re.search(r"[a-zA-Z0-9]", clean_text):
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation="Sentence contains only punctuation or symbols.",
            confidence_rationale="No semantic lexical tokens present.",
        )

    # Check 1: Statistical comparative equivalence ambiguity
    # e.g., "no significant difference in hepatotoxicity between groups"
    m_stat = RE_NO_SIGNIFICANT_DIFF.search(clean_text)
    if m_stat:
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation="Contains comparative statistical equivalence wording ('no significant difference'), which does not prove absence or presence of toxicity.",
            matched_cue=m_stat.group(0),
            confidence_rationale="Comparative statistical statements between study arms are inherently ambiguous regarding absolute toxicity.",
        )

    # Check 2: Explicit dose-dependent contrast
    # e.g., "at low doses ... but at high doses"
    m_dose = RE_DOSE_CONTRAST.search(clean_text)
    if m_dose:
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation="Contains dose-dependent contrast wording with opposing or threshold effects across concentrations.",
            matched_cue=m_dose.group(0),
            confidence_rationale="Dose-dependent differential responses cannot be simplified into binary polarity without full pharmacological modeling.",
        )

    # Check 3: Mask non-negating additive idioms ('not only ... but also', 'not just ... but')
    # to protect against naive substring negation triggers and false contrastive splitting
    masked_text = re.sub(
        r"\bnot\s+(?:only|just)\b(.+?)\bbut\s+(?:also\s+)?\b",
        r"ADDITIVE_CORRELATIVE_TOKEN \1 AND_CORRELATIVE_TOKEN ",
        clean_text,
        flags=re.IGNORECASE,
    )
    masked_text = RE_ADDITIVE_NOT_ONLY.sub("ADDITIVE_CORRELATIVE_TOKEN", masked_text)

    # Check 4: Scan for linguistic cues
    matched_uncert: List[str] = []
    for pat in UNCERTAINTY_PATTERNS:
        for m in pat.finditer(masked_text):
            matched_uncert.append(m.group(0))

    matched_neg: List[str] = []
    for pat in NEGATION_PATTERNS:
        for m in pat.finditer(masked_text):
            matched_neg.append(m.group(0))

    matched_aff: List[str] = []
    for pat in AFFIRMATIVE_PATTERNS:
        for m in pat.finditer(masked_text):
            matched_aff.append(m.group(0))

    # Check 5: Conflicting or mixed cues in the same sentence
    if matched_neg and matched_aff:
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation="Sentence contains conflicting or mixed polarity cues (both negated and affirmative wording detected).",
            matched_cue=f"Negation: '{matched_neg[0]}' | Affirmative: '{matched_aff[0]}'",
            confidence_rationale="Simultaneous presence of negated and affirmative claim predicates creates clause ambiguity.",
        )

    # Check 6: Multi-clause contrastive conjunctions with opposing clause cues
    if RE_CONTRASTIVE_CONJUNCTION.search(masked_text):
        clauses = re.split(
            r"\b(?:but|however|whereas|although|yet|nevertheless|conversely)\b",
            masked_text,
            flags=re.IGNORECASE,
        )
        if len(clauses) > 1:
            clause_has_neg = any(any(p.search(c) for p in NEGATION_PATTERNS) for c in clauses)
            clause_has_aff = any(any(p.search(c) for p in AFFIRMATIVE_PATTERNS) for c in clauses)
            if clause_has_neg and clause_has_aff:
                return PolarityAnnotation(
                    label=POLARITY_UNCERTAIN,
                    ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
                    explanation="Multi-clause sentence with contrastive conjunctions linking conflicting claims.",
                    matched_cue=f"Contrastive conjunction: '{RE_CONTRASTIVE_CONJUNCTION.search(masked_text).group(0)}'",
                    confidence_rationale="Opposing clauses cannot be resolved without dependency parsing.",
                )

    # Check 7: Hedging and Epistemological Uncertainty (preempts affirmative claims)
    if matched_uncert:
        return PolarityAnnotation(
            label=POLARITY_UNCERTAIN,
            ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
            explanation=POLARITY_EXPLANATIONS[POLARITY_UNCERTAIN],
            matched_cue=matched_uncert[0],
            confidence_rationale="Speculative or hedging language indicates findings are not asserted as established claims.",
        )

    # Check 8: Explicit Negation
    if matched_neg:
        return PolarityAnnotation(
            label=POLARITY_NEGATED,
            ui_label=POLARITY_UI_LABELS[POLARITY_NEGATED],
            explanation=POLARITY_EXPLANATIONS[POLARITY_NEGATED],
            matched_cue=matched_neg[0],
            confidence_rationale="Clear negative predicate detected without conflicting affirmative or hedging indicators.",
        )

    # Check 9: Explicit Affirmation
    if matched_aff:
        return PolarityAnnotation(
            label=POLARITY_AFFIRMATIVE,
            ui_label=POLARITY_UI_LABELS[POLARITY_AFFIRMATIVE],
            explanation=POLARITY_EXPLANATIONS[POLARITY_AFFIRMATIVE],
            matched_cue=matched_aff[0],
            confidence_rationale="Clear affirmative causation or association predicate detected without negation or hedging.",
        )

    # Check 10: Conservative Fallback
    return PolarityAnnotation(
        label=POLARITY_UNCERTAIN,
        ui_label=POLARITY_UI_LABELS[POLARITY_UNCERTAIN],
        explanation=POLARITY_EXPLANATIONS[POLARITY_UNCERTAIN],
        matched_cue=None,
        confidence_rationale="No definitive affirmative or negated relationship cue detected; conservative uncertain default applied.",
    )
