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
    return scored


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
