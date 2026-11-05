# Chronic Disease Risk Explorer

A Streamlit app for exploring educational machine learning estimates for **diabetes**, **heart disease** and **chronic kidney disease**, with each model's data, test results and limitations shown alongside it.

> **Educational only.** This project is not a medical device and has not been clinically validated. Its estimates are not a diagnosis or a treatment recommendation. Please talk to a qualified healthcare professional about your health. If you may be having an emergency, contact your local emergency services.

## What the app does

- Gives a separate estimate for each condition, using a separate model and pipeline.
- Shows which answers moved an estimate, and compares it with the rate in that dataset.
- Reports held-out test results, calibration, subgroup results and known limitations.
- Lets you leave clinical values blank where the model allows it, and warns you when it fills them in.
- Asks for no identifiers and has no accounts. Answers are not written to a file or database.

**The estimates are not forecasts.** All three datasets capture people at a single point in time, so a number describes how often similar records had the outcome in that dataset. The app has no "low" or "high" risk labels, because none of these models supports clinical categories.

## Run it locally

You need [Python 3.12](https://www.python.org/downloads/) and PowerShell on Windows. These commands install into a project-only `.venv` folder, and you don't need to change any PowerShell settings.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at <http://localhost:8501>. Trained models are included, so you don't need to download data or train anything first.

On macOS or Linux, use `python3.12 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Datasets

The datasets are not in this repository. Training downloads them to the git-ignored `data/raw/` folder and checks each file against a pinned SHA-256 checksum. All three are public, de-identified research datasets, and the app cannot display anyone's records.

| Condition | Dataset | Records | Target | License |
|---|---|---|---|---|
| Diabetes | [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) | 253,680 | Self-reported prediabetes or diabetes | [CC0, as labelled by the Kaggle curator](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset); UCI lists no license of its own |
| Heart disease | [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 Cleveland, plus 617 from other sites | Over 50% narrowing on angiography (`num` above 0) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Kidney disease | [UCI Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | 400 | The source's `ckd` or `notckd` label | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

**A note on the diabetes license:** the CC0 label comes from the person who cleaned the data on Kaggle, not from the CDC. The underlying survey is the [CDC's BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html), and that page states no specific license. If your use depends on the license, please check it yourself.

**A note on the UCI diabetes page:** it links to BRFSS 2014 and mentions 35 features, but the file matches the 21-feature BRFSS 2015 dataset described by its Kaggle curator. This project treats it as BRFSS 2015.

### Citations

- CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. <https://doi.org/10.24432/C53919>. Source survey: CDC Behavioral Risk Factor Surveillance System, 2015. Cleaned dataset curated by Alex Teboul on Kaggle.
- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>.
- Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5G020>.
- Collins, G. S., Ogundimu, E. O., & Altman, D. G. (2016). Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Statistics in Medicine*, 35(2), 214-226. <https://doi.org/10.1002/sim.6787>.

### Limitations by dataset

**Diabetes**
- Answers are self-reported, so undiagnosed diabetes counts as "no diabetes".
- Some inputs, such as poor general health, can be consequences of diabetes rather than causes.
- The file has no survey weights, so its 13.9% rate is not a US prevalence estimate.
- Type 1 and type 2 are not separated, and sex was recorded with only two options.

**Heart disease**
- Everyone had been referred for angiography in the 1980s, so results don't apply to people who haven't been referred.
- 206 of the 303 Cleveland records are male.
- In this referred group, people with no chest pain had the *highest* rate of narrowing (73.7%, compared with 15.6% to 31.6% for the other types). This pattern shouldn't be applied to anyone else.

**Kidney disease**
- 400 records from one hospital over about two months, including 19 people under 18. The app only accepts adults.
- The source doesn't document the clinical criteria behind its label.
- It has no sex or ethnicity field, so performance across those groups is unknown.
- Missing values are concentrated in CKD records, so missingness itself reveals much of the label.

## Results

These come from one run with seed 42 and are stored in [`models/`](models/). Intervals are 95% percentile bootstrap intervals on the held-out test set.

| Model | Selected | Test records (positive) | ROC AUC | PR AUC | Brier (constant-rate baseline) | Average estimate vs. actual |
|---|---|---|---|---|---|---|
| Diabetes | Gradient boosting | 50,663 (7,081) | 0.828 (0.823 to 0.832) | 0.434 (0.421 to 0.446) | 0.097 (0.120) | 13.9% vs 14.0% |
| Heart disease | Logistic regression | 61 (28) | 0.881 (0.777 to 0.964) | 0.888 (0.762 to 0.970) | 0.130 (0.248) | 43.3% vs 45.9% |
| Kidney disease | Gradient boosting | 80 (50) | 1.000 (see below) | 1.000 | 0.013 (0.234) | 64.1% vs 62.5% |

**Don't read the kidney score as an achievement.** A perfect result on 80 records is much more likely to reflect this dataset than real accuracy, and the bootstrap interval can't extend past 1.000, so it doesn't show the true uncertainty. The score stayed perfect after removing creatinine, urea and albumin. Using only *which* values were missing, and none of the values, a simple model still reached a cross-validated ROC AUC of 0.854.

**The heart model does less well elsewhere.** It was developed on Cleveland only and tested on the other sites, where most `ca` and `thal` values are missing:

| Site | Records (positive) | ROC AUC | Average estimate vs. actual |
|---|---|---|---|
| Hungary | 294 (106) | 0.887 (0.843 to 0.926) | 33.7% vs 36.1% |
| VA Long Beach | 200 (149) | 0.700 (0.614 to 0.778) | 48.2% vs 74.5% |
| Switzerland | 123 (115), too few negatives for a stable estimate | 0.749 (0.500 to 0.940) | 51.5% vs 93.5% |

## How the models were built

Each condition has its own module in [`chronic_risk/pipelines/`](chronic_risk/pipelines/) covering its inputs, cleaning, subgroups and documentation.

1. **Clean:** repairs each file's specific problems, such as stray tabs in the kidney labels and `chol = 0` placeholders in the heart data.
2. **Split:** sets aside 20% as the test set before anything else. Rows with identical feature values stay on one side of every split, which matters because only 183,096 of the 253,680 diabetes rows are unique on the model's inputs.
3. **Preprocess:** median and most-frequent imputation, scaling and one-hot encoding all sit inside a scikit-learn `Pipeline`, so they are fitted only on training rows. A test checks this.
4. **Select:** compares logistic regression and gradient boosting candidates with grouped cross-validation on training rows, and picks the simplest within one standard error of the best log loss.
5. **Calibrate, if justified:** see below.
6. **Test once:** reports ROC AUC, PR AUC, Brier score, log loss, calibration slope, a calibration table, subgroup results and a constant-rate baseline.

### Calibration

Post-hoc calibration needs enough data to fit and to judge it. This project's rule is a dedicated split with at least 200 positive and 200 negative records. That threshold is a project choice, informed by [Collins et al. (2016)](https://doi.org/10.1002/sim.6787), who recommend at least 100, and ideally 200, events for validating a model. It isn't a rule from that paper.

| Model | Eligible? | Result |
|---|---|---|
| Diabetes | Yes (7,052 positive, 43,883 negative) | Sigmoid and isotonic were compared out of fold on the calibration split. **Not adopted**, because the Brier improvement's 95% interval included zero (-0.00002 to 0.00005). Test calibration slope: 0.987. |
| Heart disease | No, too few records | Not separately calibrated. The app says so. |
| Kidney disease | No, too few records | Not separately calibrated. The app says so. |

The test set played no part in these decisions. When diabetes calibration wasn't adopted, the rows set aside for it stayed unused in the final fit, so the model reported is exactly the one evaluated.

### Variables left out on purpose

- **Diabetes:** income and education, to collect less personal data and avoid proxies for social status. Healthcare access questions, because they describe access rather than health.
- **Kidney:** creatinine, urea and albumin, because they help define the diagnosis. Red and white cell counts, red cell status, potassium and sodium, because each is more than 20% missing. Urine microscopy findings, to stick to more commonly available results.

### Cutoffs

The Models & data page shows what five cutoffs would do, purely to illustrate the trade-off. They are labelled as illustrative, and none is medically established.

## Privacy

- The app asks for no identifiers. Tests check that no input does, and that runtime code has no file-writing or logging calls.
- While a session is open, answers stay in the app server's memory so the form works. They aren't written to disk, and **Clear answers** removes them sooner.
- Framework usage statistics are turned off, and error details are hidden from the browser.
- A third-party host has its own terms and infrastructure logging. Check them before relying on a hosted copy.
- The app needs no secrets or API keys.

## Retrain the models

```powershell
.\.venv\Scripts\python.exe -m chronic_risk.train all
```

This downloads the datasets if needed, then writes `models/*.joblib`, reports and a checksum manifest. You can also pass `diabetes`, `heart` or `kidney`.

The app refuses to load a model that fails its checksum or was built with a different scikit-learn version. **Only load `.joblib` files you trust**, because they can run code when loaded, and the checksum only detects changes relative to the manifest beside it.

## Run the checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Tests use artificial data and the included models, so they don't need the datasets. GitHub Actions runs the same checks on each push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository. Check `git status` first, and make sure `data/` and `.venv/` aren't listed.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) and select **Create app**.
3. Choose your repository and branch, and set the main file path to `app.py`.
4. Under **Advanced settings**, choose **Python 3.12**, which the models were tested with. You don't need to add any secrets.
5. Select **Deploy**.

**Check who can see it.** According to [Streamlit's documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app), an app from a public repository is public by default, and one from a private repository is private to the workspace's developers. You can have one private app at a time. Change this under the app's sharing settings.

## Project layout

```text
app.py                     Entry point and navigation
app_pages/                 One script per page
chronic_risk/
  pipelines/               Diabetes, heart and kidney: inputs, cleaning, documentation
  data.py                  Verified dataset downloads
  modeling.py              Splits, candidates, model selection, calibration
  evaluation.py            Metrics (aggregates only)
  train.py                 Training command
  inference.py             Verified model loading, validation, estimates
  ui/                      Shared page, styling and chart code
models/                    Trained models, reports and checksum manifest
tests/                     Test suite (no real data needed)
.streamlit/config.toml     Theme and privacy settings
```

## License

No license has been chosen for this project's code yet, so default copyright applies until a `LICENSE` file is added. The datasets keep their own terms, listed above, and CC BY 4.0 requires the attribution given in [Citations](#citations).
