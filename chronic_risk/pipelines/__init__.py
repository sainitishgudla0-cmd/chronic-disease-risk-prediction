"""The independent pipeline for each condition."""

from chronic_risk.pipelines.diabetes import PIPELINE as DIABETES
from chronic_risk.pipelines.heart import PIPELINE as HEART
from chronic_risk.pipelines.kidney import PIPELINE as KIDNEY

PIPELINES = {p.key: p for p in (DIABETES, HEART, KIDNEY)}
