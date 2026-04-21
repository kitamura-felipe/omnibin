"""
CRIMSON score calculation, extracted verbatim from
https://github.com/rajpurkarlab/CRIMSON/blob/main/CRIMSON/generate_score.py
(the `CRIMSONScore._calculate_crimson` method), MIT-licensed.

Copied byte-for-byte so the scalar matches the upstream package. The only
change vs. upstream is hoisting the method into a module-level function;
its logic (weights, formulas, return shape) is untouched.

© rajpurkarlab — see LICENSE in this directory.
"""
from __future__ import annotations


def calculate_crimson(evaluation: dict) -> dict:
    """Calculate CRIMSON score from an LLM evaluation dict.

    Args:
        evaluation: parsed JSON from the judge model, with keys
            `reference_findings`, `predicted_findings`, `matched_findings`,
            and `errors`.

    Returns:
        Dictionary with `raw_evaluation`, `error_counts`, `weighted_error_counts`,
        `metrics`, and `crimson_score` (rounded to 4 decimals, in (-1, 1]).
    """
    errors = evaluation.get("errors", {})
    matched = evaluation.get("matched_findings", [])
    reference_findings_list = evaluation.get("reference_findings", [])
    predicted_findings_list = evaluation.get("predicted_findings", [])

    # Weight mapping for clinical significance
    significance_weights = {
        "urgent": 1.0,
        "actionable_not_urgent": 0.5,
        "not_actionable_not_urgent": 0.25,
        "benign_expected": 0.0,
    }

    # Weight mapping for attribute error severity
    attribute_severity_weights = {
        "significant": 0.5,
        "negligible": 0.0,
    }

    def calculate_weighted_count(error_list, weights=significance_weights, key="clinical_significance"):
        """Calculate weighted count based on significance/severity."""
        return sum(weights.get(error.get(key, ""), 0.25) for error in error_list)

    ref_weight_by_id = {
        ref["id"]: significance_weights.get(ref.get("clinical_significance", ""), 0.25)
        for ref in reference_findings_list
    }
    pred_weight_by_id = {
        pred["id"]: significance_weights.get(pred.get("clinical_significance", ""), 0.25)
        for pred in predicted_findings_list
    }

    E_false = sum(pred_weight_by_id.get(f_id, 0.0) for f_id in errors.get("false_findings", []))

    E_miss = sum(ref_weight_by_id.get(m_id, 0.0) for m_id in errors.get("missing_findings", []))

    attr_errors = errors.get("attribute_errors", [])
    n_location = sum(1 for e in attr_errors if "location" in e.get("error_types", []))
    n_severity = sum(1 for e in attr_errors if "severity" in e.get("error_types", []))
    n_descriptor = sum(1 for e in attr_errors if "descriptor" in e.get("error_types", []))
    n_measurement = sum(1 for e in attr_errors if "measurement" in e.get("error_types", []))
    n_certainty = sum(1 for e in attr_errors if "certainty" in e.get("error_types", []))
    n_unspecific = sum(1 for e in attr_errors if "unspecific" in e.get("error_types", []))
    n_overinterpretation = sum(1 for e in attr_errors if "overinterpretation" in e.get("error_types", []))
    n_temporal = sum(1 for e in attr_errors if "temporal" in e.get("error_types", []))

    attr_errors_by_ref_id = {}
    for err in attr_errors:
        ref_id = err["ref_id"]
        if ref_id not in attr_errors_by_ref_id:
            attr_errors_by_ref_id[ref_id] = []
        attr_errors_by_ref_id[ref_id].append(err)

    N_G = calculate_weighted_count(reference_findings_list)
    if N_G == 0 and not reference_findings_list:
        N_G = len(matched) + E_miss

    E_penalty = E_false

    # Weighted sum of matched findings with partial credit for attribute errors
    matched_ref_ids = set()
    correct = 0.0
    for m in matched:
        ref_id = m["ref_id"]
        if ref_id in matched_ref_ids:
            continue  # Already counted this reference finding
        matched_ref_ids.add(ref_id)
        base_weight = ref_weight_by_id.get(ref_id, 0.0)

        finding_attr_errors = attr_errors_by_ref_id.get(ref_id, [])

        if not finding_attr_errors:
            correct += base_weight
        else:
            sum_error_weights = sum(
                attribute_severity_weights.get(err.get("severity", ""), 0.25) for err in finding_attr_errors
            )
            credit_factor = base_weight / (base_weight + sum_error_weights) if (base_weight + sum_error_weights) > 0 else 0.0
            correct += base_weight * credit_factor

    errors_more_than_correct = E_penalty - correct

    if N_G == 0:
        S = 1.0 if E_penalty == 0 and E_miss == 0 else -(E_penalty + E_miss + 1)
    else:
        S = (correct - E_penalty) / N_G

    if S >= 0:
        crimson = S
    else:
        if errors_more_than_correct > 0:
            crimson = -1 * errors_more_than_correct / (1 + errors_more_than_correct)
        else:
            crimson = 0

    return {
        "raw_evaluation": evaluation,
        "error_counts": {
            "false_findings": len(errors.get("false_findings", [])),
            "missing_findings": len(errors.get("missing_findings", [])),
            "attribute_errors": len(attr_errors),
            "location_errors": n_location,
            "severity_errors": n_severity,
            "descriptor_errors": n_descriptor,
            "measurement_errors": n_measurement,
            "certainty_errors": n_certainty,
            "unspecific_errors": n_unspecific,
            "overinterpretation_errors": n_overinterpretation,
            "temporal_errors": n_temporal,
        },
        "weighted_error_counts": {
            "false_findings": E_false,
            "missing_findings": E_miss,
            "attribute_errors": calculate_weighted_count(attr_errors, attribute_severity_weights, "severity"),
        },
        "metrics": {
            "N_G": N_G,
            "E_penalty": E_penalty,
            "correct": correct,
            "errors_more_than_correct": errors_more_than_correct,
            "S": S,
        },
        "crimson_score": round(crimson, 4),
    }
