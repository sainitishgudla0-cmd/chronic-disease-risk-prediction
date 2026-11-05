"""Plain-language evaluation results for each model, read directly from the training reports."""

import pandas as pd
import streamlit as st

from chronic_risk.inference import ArtifactError, load_report
from chronic_risk.pipelines import PIPELINES
from chronic_risk.ui import charts, theme

PERCENT = st.column_config.NumberColumn(format="percent")
THREE_DP = st.column_config.NumberColumn(format="%.3f")

st.title("Models & data")
theme.lede(
    "How each model was built and tested, and where it falls short. All numbers come from "
    "training reports, using test records the model never saw during training."
)

choice = st.radio(
    "Condition",
    list(PIPELINES),
    format_func=lambda key: PIPELINES[key].title,
    horizontal=True,
    key="models_condition",
)
pipeline = PIPELINES[choice]

try:
    report = load_report(choice)
except ArtifactError as error:
    st.error(f"Results are unavailable right now. {error}")
    st.stop()

test = report["test"]
metrics, ci = test["metrics"], test["ci95"]


def _interval(name: str) -> str:
    return f"95% interval {ci[name][0]:.3f} to {ci[name][1]:.3f}"


st.subheader("Data")
st.markdown(
    f"**[{pipeline.dataset_name}]({pipeline.dataset_url})** · License: "
    f"[{pipeline.license_name}]({pipeline.license_url})"
)
st.markdown(f"**People in the data:** {pipeline.population}")
st.markdown(f"**What the outcome means:** {pipeline.target_definition}")
counts = report["counts"]
st.markdown(
    f"**Size:** {counts['total']:,} records, of which {counts['fit']:,} fitted the model and "
    f"{counts['test']:,} were held out for testing. "
    f"Positive rate in development data: {counts['development_positive_rate']:.1%}."
)

st.subheader("Limitations")
for item in pipeline.limitations:
    st.markdown(f"- {item}")
if pipeline.excluded_features:
    with st.expander("Variables deliberately left out"):
        for names, reason in pipeline.excluded_features.items():
            st.markdown(f"- **{names}:** {reason}")

st.subheader("Test results")
if choice == "kidney":
    theme.notice(
        "A perfect score on a small test set should raise suspicion rather than confidence. The "
        "interval cannot extend past perfect, so it doesn't show the true uncertainty, and this "
        "dataset's unusually clean separation is unlikely to hold in everyday care.",
        urgent=True,
    )
first, second, third = st.columns(3)
first.metric(
    "ROC AUC",
    f"{metrics['roc_auc']:.3f}",
    help="How well the model ranks people with the outcome above people without it. "
    "0.5 is chance, 1.0 is perfect ranking.",
)
first.caption(_interval("roc_auc"))
second.metric(
    "Brier score",
    f"{metrics['brier']:.3f}",
    help="Average squared error of the probabilities. Lower is better.",
)
second.caption(
    f"{_interval('brier')}. Always guessing the average rate scores "
    f"{report['baseline_test']['brier']:.3f}."
)
third.metric(
    "Average estimate vs. actual",
    f"{test['mean_predicted']:.1%} vs {test['observed_rate']:.1%}",
    help="Close values mean the model is about right on average.",
)
third.caption(f"{test['n']:,} test records, {test['n_positive']:,} with the outcome.")

left, right = st.columns(2)
with left:
    st.markdown("**Do estimates match what happened?**")
    st.altair_chart(charts.calibration_chart(report["calibration_table"]), width="stretch")
    st.caption(
        "Test records grouped by estimate. Points near the dashed line mean estimates matched "
        "what happened. Bars show 95% intervals for each group."
    )
with right:
    st.markdown("**How well does it separate the two groups?**")
    st.altair_chart(charts.roc_chart(report["roc"]), width="stretch")
    st.caption(
        "Curves closer to the top-left separate better. The dashed line is what chance alone "
        "would produce."
    )

st.subheader("Calibration")
calibration = report["calibration"]
if calibration["adopted"]:
    st.markdown(
        f"Estimates were adjusted with **{calibration['method']} calibration**. It was fitted on "
        f"a separate split ({calibration['calibration_split_positive']:,} positive and "
        f"{calibration['calibration_split_negative']:,} negative records). It was adopted because "
        "it lowered the Brier score out of fold with a 95% interval above zero. The test set "
        "played no part in that decision."
    )
elif calibration["eligible"]:
    st.markdown(
        f"Calibration was tried on a separate split "
        f"({calibration['calibration_split_positive']:,} positive and "
        f"{calibration['calibration_split_negative']:,} negative records) but not adopted, "
        "because it did not clearly improve the Brier score out of fold. Estimates are the "
        "model's direct outputs, and the chart above shows how they compared with test outcomes."
    )
else:
    st.markdown(
        f"**These estimates are not separately calibrated.** This project attempts calibration "
        f"only with at least {calibration['minimum_per_class']} positive and "
        f"{calibration['minimum_per_class']} negative records in a dedicated split, and this "
        "dataset is too small for that. Estimates are the model's direct outputs and could be "
        "too high or too low, as the chart above suggests."
    )

st.subheader("Results by group")
st.dataframe(
    pd.DataFrame(report["subgroups"]).rename(
        columns={
            "group": "Group",
            "n": "Records",
            "n_positive": "With outcome",
            "observed_rate": "Actual rate",
            "mean_predicted": "Average estimate",
            "roc_auc": "ROC AUC",
            "brier": "Brier score",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Actual rate": PERCENT,
        "Average estimate": PERCENT,
        "ROC AUC": THREE_DP,
        "Brier score": THREE_DP,
    },
)
st.caption(
    "'None' in the AUC and Brier columns means the group had fewer than 10 records in one outcome "
    "class, too few to report. Small groups have uncertain results."
)

if "external" in report:
    st.subheader("Testing at other hospitals")
    st.markdown(
        "The model was developed only on Cleveland records and then tested on three other sites, "
        "where most specialist imaging values are missing and had to be filled in."
    )
    external = report["external"]
    chart_rows = [
        {
            "site": "Cleveland held-out test",
            "auc": metrics["roc_auc"],
            "low": ci["roc_auc"][0],
            "high": ci["roc_auc"][1],
            "n": test["n"],
        }
    ] + [
        {
            "site": site,
            "auc": item["metrics"]["roc_auc"],
            "low": item["ci95"]["roc_auc"][0],
            "high": item["ci95"]["roc_auc"][1],
            "n": item["n"],
        }
        for site, item in external.items()
    ]
    st.altair_chart(charts.site_auc_chart(chart_rows), width="stretch")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Data": site,
                    "Records": item["n"],
                    "Actual rate": item["observed_rate"],
                    "Average estimate": item["mean_predicted"],
                    "ROC AUC": item["metrics"]["roc_auc"],
                    "Brier score": item["metrics"]["brier"],
                    "Stable estimate?": "No, too few in one class"
                    if item["too_few_for_stable_estimate"]
                    else "Yes",
                }
                for site, item in external.items()
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Actual rate": PERCENT,
            "Average estimate": PERCENT,
            "ROC AUC": THREE_DP,
            "Brier score": THREE_DP,
        },
    )
    st.caption(
        "When 'Average estimate' is well below 'Actual rate', the model underestimates at that "
        "site. Performance varies by location, so results from one hospital may not carry over "
        "to another."
    )

diagnostics = report.get("diagnostics", {})
if "positive_rate_by_chest_pain" in diagnostics:
    st.subheader("Why 'no chest pain' raises the estimate")
    st.markdown(
        "Everyone here had been referred for a heart investigation, and in this data, people with "
        "no chest pain had the highest rate of narrowing. This pattern reflects this referred "
        "group and should not be applied to anyone else."
    )
    st.dataframe(
        pd.DataFrame(diagnostics["positive_rate_by_chest_pain"]).rename(
            columns={"chest_pain_type": "Chest pain type", "n": "Records", "rate": "Rate"}
        ),
        hide_index=True,
        width="stretch",
        column_config={"Rate": PERCENT},
    )

if "missingness_only_cv_auc" in diagnostics:
    st.subheader("Leakage checks")
    complete = diagnostics["complete_case_test"]
    complete_auc = "too few to report" if complete["auc"] is None else f"{complete['auc']:.3f}"
    st.markdown(
        f"""
- **Missingness alone:** using only *which* values are absent, and none of the values, a simple
  model reaches a cross-validated ROC AUC of **{diagnostics["missingness_only_cv_auc"]:.3f}** on
  development rows. Missingness itself reveals much of the label.
- **All 24 original variables:** a model including creatinine, urea and albumin scores a test ROC
  AUC of **{diagnostics["all_24_features_test_auc"]:.3f}**. The app's model leaves those out.
- **Rows with no filled-in values:** ROC AUC **{complete_auc}**, from
  {complete["n_positive"]} positive and {complete["n_negative"]} negative test rows.
"""
    )
    st.caption(
        "Even after these safeguards the separation remains nearly perfect, which suggests the "
        "result says more about how this dataset was assembled than about real-world accuracy."
    )

with st.expander("How the model was chosen"):
    st.markdown(
        f"Candidates were compared with grouped cross-validation "
        f"({report['cv']['folds']} folds, {report['cv']['repeats']} repeat(s)) on training rows "
        "only. The simplest candidate within one standard error of the best log loss was chosen. "
        f"Selected: **{report['selected_model']}**."
    )
    st.dataframe(
        pd.DataFrame(report["candidate_comparison"]).rename(
            columns={
                "model": "Model",
                "log_loss": "Log loss",
                "log_loss_se": "Standard error",
                "brier": "Brier score",
                "roc_auc": "ROC AUC",
                "selected": "Selected",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with st.expander("What would different cutoffs do?"):
    st.markdown(
        "**These cutoffs are illustrative and none is medically established.** They only show the "
        "trade-off: flagging more people finds more true cases but also more false alarms."
    )
    st.dataframe(
        pd.DataFrame(report["cutoffs"]).rename(
            columns={
                "cutoff": "Illustrative cutoff",
                "share_flagged": "Share flagged",
                "sensitivity": "True cases found",
                "specificity": "Non-cases cleared",
                "ppv": "Flagged who were true cases",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Illustrative cutoff": PERCENT,
            "Share flagged": PERCENT,
            "True cases found": PERCENT,
            "Non-cases cleared": PERCENT,
            "Flagged who were true cases": PERCENT,
        },
    )
