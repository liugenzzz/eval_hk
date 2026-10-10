"""Industrial description scoring reuses execution, with an independent rubric."""

from __future__ import annotations

import pandas as pd

from ..compliance import TEXT
from ..industrial_rubric import INDUSTRIAL_PROMPT_MARKER
from . import JUDGE, ScoringContext, register
from .judge_text import score_judge_text


@register("industrial_describe", engine=JUDGE, pairwise=False, answer_form=TEXT)
def score_industrial(data, ctx: ScoringContext):
    if INDUSTRIAL_PROMPT_MARKER not in ctx.judge_client.settings.pointwise_prompt:
        raise ValueError("industrial_describe requires the industrial_v1 judge prompt")
    scored = score_judge_text(data, ctx)
    if "quality_score" not in scored:
        # Pointwise can be disabled for an offline diagnostics-only run.
        scored["quality_score"] = None
    if ctx.do_pointwise:
        validate_frozen_reference_scores(scored)
    return scored


def validate_frozen_reference_scores(data: pd.DataFrame) -> None:
    """A published usable-only set must retain its fixed scored denominator."""
    marker = "meta.reference_validity"
    if marker not in data:
        return
    frozen = data[marker].eq("usable")
    if not frozen.any():
        return
    valid = (pd.to_numeric(data.get("quality_score"), errors="coerce").notna()
             & data.get("industrial_reference_status", pd.Series(index=data.index, dtype=object)).eq("usable"))
    failed = data.loc[frozen & ~valid]
    if not failed.empty:
        ids = failed["index"].astype(str).tolist() if "index" in failed else failed.index.astype(str).tolist()
        raise ValueError(
            "frozen industrial reference denominator changed for sample IDs "
            + ", ".join(ids[:20])
            + (f" (and {len(ids) - 20} more)" if len(ids) > 20 else "")
            + "; retry judge failures or review disputed references; do not fill scores with zero"
        )


def validate_scored_industrial(data: pd.DataFrame) -> None:
    """An old equipment score file cannot stand in for industrial rubric results."""
    required = {"rubric", "industrial_reference_status", "object_score", "defect_score",
                "appearance_score", "location_score", "industrial_critical_errors", "quality_score"}
    if not required.issubset(data.columns):
        raise ValueError("industrial scored input requires industrial_v1 columns; rerun from predictions")
    tags = data["rubric"].dropna().astype(str)
    if tags.empty or not tags.eq("industrial_v1").all():
        raise ValueError("industrial scored input must use rubric industrial_v1")
    scored = pd.to_numeric(data["quality_score"], errors="coerce").notna()
    if not data.loc[scored, "rubric"].eq("industrial_v1").all():
        raise ValueError("industrial scored rows have missing rubric tags")
    validate_frozen_reference_scores(data)
