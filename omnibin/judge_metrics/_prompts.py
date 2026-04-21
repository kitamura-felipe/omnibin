"""
Verbatim prompt templates from the original GREEN and RadFact repositories.

Kept in a separate module so API-mode scorers can replay each paper's
pipeline with a different judge LLM without re-implementing the metric.

Sources:
- GREEN (Apache-2.0):
    https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/utils.py
    https://github.com/Stanford-AIMI/GREEN/blob/main/green_score/green.py
- RadFact (MIT):
    https://github.com/microsoft/RadFact/blob/main/src/radfact/llm_utils/report_to_phrases/prompts/cxr/system_message.txt
    https://github.com/microsoft/RadFact/blob/main/src/radfact/llm_utils/nli/prompts/cxr/system_message_ev_singlephrase.txt

When the corresponding upstream package is installed, the original
implementation is used and this module is bypassed.
"""
from __future__ import annotations


# Truncation for GREEN inputs. The shipped judge model was trained with
# reports capped at 300 whitespace tokens each.
GREEN_MAX_TOKENS_PER_REPORT = 300


def green_truncate(text: str, max_len: int = GREEN_MAX_TOKENS_PER_REPORT) -> str:
    """Reproduce GREEN's whitespace-token truncation."""
    return " ".join(text.split()[:max_len])


def green_build_prompt(reference: str, candidate: str) -> str:
    """
    Build the GREEN judge prompt verbatim from green_score/utils.py::make_prompt.

    The 4-space indentation and exact wording are preserved — the published
    model was trained on this formatting and keeping it unchanged gives
    downstream LLMs the best chance of producing parseable output.
    """
    text1 = green_truncate(reference)
    text2 = green_truncate(candidate)
    prompt = f"""Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

    Process Overview: You will be presented with:

    1. The criteria for making a judgment.
    2. The reference radiology report.
    3. The candidate radiology report.
    4. The desired format for your assessment.

    1. Criteria for Judgment:

    For each candidate report, determine:

    The count of clinically significant errors.
    The count of clinically insignificant errors.

    Errors can fall into one of these categories:

    a) False report of a finding in the candidate.
    b) Missing a finding present in the reference.
    c) Misidentification of a finding's anatomic location/position.
    d) Misassessment of the severity of a finding.
    e) Mentioning a comparison that isn't in the reference.
    f) Omitting a comparison detailing a change from a prior study.
    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.

    2. Reference Report:
    {text1}

    3. Candidate Report:
    {text2}

    4. Reporting Your Assessment:

    Follow this specific format for your output, even if no errors are found:
    ```
    [Explanation]:
    <Explanation>

    [Clinically Significant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

    [Clinically Insignificant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

    [Matched Findings]:
    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
    ```
"""
    return prompt


# The GREEN error subcategory labels, in order a..f.
GREEN_ERROR_SUBCATEGORIES = (
    "False report of a finding in the candidate",
    "Missing a finding present in the reference",
    "Misidentification of a finding's anatomic location/position",
    "Misassessment of the severity of a finding",
    "Mentioning a comparison that isn't in the reference",
    "Omitting a comparison detailing a change from a prior study",
)


# ---------------------------------------------------------------------------
# RadFact verbatim system messages
# ---------------------------------------------------------------------------

RADFACT_DECOMPOSITION_SYSTEM = """You are an AI radiology assistant. You are helping process reports from chest X-rays.

Please extract phrases from the radiology report which refer to objects, findings, or anatomies visible in a chest X-ray, or the absence of such.

Rules:
- If a sentence describes multiple findings, split them up into separate sentences.
- Exclude clinical speculation or interpretation (e.g. "... highly suggestive of pneumonia").
- Exclude recommendations (e.g. "Recommend a CT").
- Exclude comments on the technical quality of the X-ray (e.g. "there are low lung volumes").
- Include mentions of change (e.g. "Pleural effusion has increased") because change is visible when we compare two X-rays.
- If consecutive sentences are closely linked such that one sentence can't be understood without the other one, process them together.

The objective is to extract phrases which refer to things which can be located on a chest X-ray, or confirmed not to be present."""


RADFACT_ENTAILMENT_SYSTEM = (
    "You are an AI radiology assistant. Your task is to assess whether a "
    "statement about a chest X-ray (the \"hypothesis\") is true or not, "
    "given a reference report about the chest X-ray. This task is known as "
    "entailment verification. If the statement is true (\"entailed\") "
    "according to the reference, provide the evidence to support it."
)


# Output-format instruction appended to RadFact prompts when driving
# non-default LLMs. RadFact's upstream uses 10-shot YAML exemplars to teach
# output format; for API mode we ask for JSON directly since modern models
# follow schema instructions reliably. This is the one documented deviation
# from the original pipeline and is surfaced to users as a disclaimer.
RADFACT_DECOMPOSITION_OUTPUT_INSTRUCTION = """Return ONLY a JSON object with this exact schema — no prose, no markdown:
{"phrases": ["phrase 1", "phrase 2", ...]}
Each phrase should be a short, self-contained statement."""


RADFACT_ENTAILMENT_OUTPUT_INSTRUCTION = """Return ONLY a JSON object with this exact schema — no prose, no markdown:
{"status": "entailment" | "not_entailment", "evidence": ["supporting phrase from reference", ...]}
Use "entailment" only if the hypothesis is supported by at least one reference phrase. Use "not_entailment" otherwise (contradicted OR unsupported). When status is "not_entailment", evidence should be an empty list."""
