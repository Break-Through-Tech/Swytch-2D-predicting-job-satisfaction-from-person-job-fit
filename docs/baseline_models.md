# Baseline Models Documentation

**Issue:** #11 (Task #5: Create documentation for baseline models)
**Milestone:** Baseline Models & Evaluation Setup
**Status:** Draft

This document explains what our team did for the October baseline models, what we found, and what we plan to do next. Every number comes from the notebooks and result files in this repository. Section 11 lists those files.

## 1. Goal

Our project asks whether a person's job satisfaction can be predicted from **person-job fit**, which means how well what a person values matches what their job offers.

A **baseline model** is a simple first model that gives us a score to beat. Ours use only **background factors** such as income, working hours, and demographics (age, sex, and similar facts about a person). In November, we will add fit information and check whether the score improves.

## 2. Data

| Dataset | Size | Used for |
|---|---|---|
| **NSCG 2023** (National Survey of College Graduates) | 94,606 people in the raw file | Job satisfaction, background factors, and personal work values |
| **Cleaned NSCG file** (`nscg2023_clean.csv`) | 79,800 employed people | Main baseline models |
| **O\*NET 30.2** (a database that describes what each occupation offers) | 6 Work Values for each occupation | What the job offers |
| **First linked group** | 6,849 people, 25 occupations | Baseline models on people linked to O\*NET |
| **Expanded linked group** | 74,016 people, 123 occupation categories | Prepared for November; no baseline results yet |

**What we predict:** The survey asks about job satisfaction on a 4-point scale. We turned this into two groups. People who answered "somewhat dissatisfied" or "very dissatisfied" are **not satisfied**, and they make up **9.9%** of the cleaned data. The other 90.1% are satisfied.

## 3. Steps We Took

1. **Compared data sources.** The team reviewed GSS, Gallup, and NLSY97 alongside NSCG.
2. **Explored NSCG (EDA).** EDA, or exploratory data analysis, means studying data before modeling. We found that most people are satisfied, so the two groups are very different in size.
3. **Cleaned NSCG.** We kept the 79,800 employed people who answered the job satisfaction question. We also converted the survey's special codes to missing values. For example, `L` means "logical skip", or a question that did not apply to that person.
4. **Linked NSCG to O\*NET.** NSCG uses its own occupation codes, so we built a **crosswalk**, a table that translates one set of codes into another. The first version linked 25 occupations. Pull request #17 later expanded it to 123 occupation categories.
5. **Prepared the features.** A **feature** is an input column for a model. We used the log of salary because a few very high salaries stretch the scale. We **standardized** numbers (put them on a similar scale) and **one-hot encoded** categories (made one yes/no column for each category).
6. **Trained the baseline models** using only background factors.
7. **Set up a fair evaluation**, because accuracy alone is misleading for our data (Section 5).

### Baseline features

| Type | Features |
|---|---|
| Numbers | Log salary, weekly hours, age |
| Categories | Sex, race/ethnicity, degree, employer sector, employer size, supervises others, marital status |

We left these out of the main baseline on purpose:

- **Personal work values and O\*NET data**, because they are saved for the November fit test.
- **`OCEDRLP`** (how related a job is to a person's degree), because it is already a simple measure of fit. We tested it separately.
- **`SAT` columns** (satisfaction with each part of a job), because they ask almost the same question as our target. Using them would be **data leakage**, which means giving the model part of the answer.

## 4. Models We Tried

| Model | Simple explanation | Why we tried it |
|---|---|---|
| **Majority class** | Always predicts "satisfied" | Shows the score of a useless model |
| **Logistic regression** | Gives each feature a weight and adds them up | Simple and easy to interpret |
| **Random forest** | Combines many decision trees | Can capture how features work together, such as salary and sector |
| **Gradient boosting** | Builds trees one by one, each correcting earlier mistakes | Usually strong on table data and fast on 80,000 rows |
| **K-nearest neighbors (KNN)** | Looks at the most similar people to make a prediction | Part of our model comparison |

## 5. How We Measured Results Fairly

The satisfied group is about 9 times larger than the not satisfied group. A model that always says "satisfied" gets 90.1% accuracy but finds nobody who is unhappy. So we did the following:

- **Stratified splits:** Every train and test piece keeps the same 9.9% rate. We used 80% of the data for training and 20% for testing.
- **Cross-validation:** The training data is split into 5 parts, and each part takes a turn being the check set.
- **Class weights:** Each not satisfied person counts more during training, so the model cannot ignore this group. KNN cannot use class weights.
- **Training-only threshold:** The **threshold** is the score at which a person gets flagged as at risk. We chose it using training data only.
- **Survey weight check:** Survey weights adjust the results so they better represent the U.S. college-graduate workforce.

### Metrics

| Metric | Meaning | Score of a useless model |
|---|---|---|
| **ROC AUC** | How often a random not satisfied person gets a higher risk score than a random satisfied person | 0.5 |
| **PR AUC** | How precise the model stays as it finds more not satisfied people | 0.099 |
| **Balanced accuracy** | Average of the correct rate for each group, so both groups count equally | 0.5 |
| **Precision** | Of the people flagged, the share who are really not satisfied | n/a |
| **Recall** | Of the people who are really not satisfied, the share the model found | n/a |
| **95% confidence interval (CI)** | The range where the true score probably falls | n/a |

## 6. Results

### All 79,800 people (test set of 15,960)

| Model | ROC AUC [95% CI] | PR AUC | Balanced accuracy | Recall | Accuracy |
|---|---|---|---|---|---|
| Majority class | 0.500 | 0.099 | 0.500 | 0.000 | 0.901 |
| Logistic regression | 0.626 [0.612, 0.640] | 0.147 | 0.594 | 0.577 | 0.607 |
| KNN | 0.634 [0.620, 0.648] | 0.157 | 0.596 | 0.638 | 0.563 |
| Random forest | 0.651 [0.636, 0.664] | 0.166 | 0.610 | 0.584 | 0.630 |
| Gradient boosting | 0.652 [0.637, 0.667] | 0.170 | 0.613 | 0.530 | 0.680 |

**Main findings:**

- **Background factors predict dissatisfaction better than chance, but only modestly.** These models are useful for ranking risk, not for labeling one person. Gradient boosting finds 53% of not satisfied people, but only 16% of the people it flags are really not satisfied.
- **There is no single best model.** Random forest and gradient boosting are statistically tied. With survey weights, random forest (0.677), KNN (0.676), and gradient boosting (0.672) are effectively tied, and logistic regression is last (0.663).
- **`OCEDRLP` helps.** Adding it raised AUC by about 0.02 (logistic regression to 0.652, gradient boosting to 0.668).
- **Personal work values help KNN.** In the KNN notebook, adding them raised AUC from 0.652 to 0.680, a gain of 0.028 [0.018, 0.038].

The KNN baseline of 0.652 is different from the 0.634 in the table because the KNN notebook's baseline includes `OCEDRLP`. Please do not compare these two numbers directly.

### Three properties we checked

| Property | Result | Evidence (gradient boosting) |
|---|---|---|
| **Accuracy:** better than chance on fair metrics | Passes | The lowest end of the CI is 0.637 for ROC AUC, which is above 0.5 |
| **Satisfaction:** risk scores follow real satisfaction | Passes | Dissatisfaction rises from 3.9% in the lowest-risk 10% of people to 20.7% in the highest-risk 10% |
| **Group correlation:** works across groups and tracks group differences | Passes, with caveats | AUC is 0.60 to 0.67 within groups. Predicted risk and real dissatisfaction across occupations have a correlation of 0.72 |

**Caveat on groups:** At one shared threshold, the share of people flagged differs across groups much more than real dissatisfaction does. For example, 13.2% of Black respondents and 9.2% of White respondents are dissatisfied, but the model flags 54.6% and 26.0%. For women and men, dissatisfaction is 10.5% and 9.3%, but the flag rates are 38.7% and 27.4%. A model like this would need a review of thresholds for each group before anyone used it to act on individuals.

### People linked to O\*NET

| Notebook | People | How it was tested | Result (ROC AUC) |
|---|---|---|---|
| Baseline evaluation | 6,849 | One test set, full background features | Logistic regression 0.623, random forest 0.624, gradient boosting 0.592 |
| Logistic regression | 6,849 | Occupations kept separate, fewer features | 0.577 |
| Random forest | 6,849 | Occupations kept separate, fewer features | 0.586 |
| KNN (full features) | 6,578 | One test set | Baseline 0.626, + work values 0.668, + O\*NET 0.630, + fit 0.634 |
| KNN (subset file only) | 6,849 | One test set, fewer features | Baseline 0.617, + work values 0.577, + O\*NET 0.562, + fit 0.562 |

These rows use different people, features, or testing methods, so they should not be ranked against each other.

O\*NET and fit features did not improve KNN on this small group. This does **not** prove that fit is unimportant. With only 25 occupations, O\*NET adds little variation, and the small test set makes small changes hard to see. The evaluation notebook estimates that a gain under about 0.05 AUC cannot be detected from one test split.

## 7. Crosswalk Creation

- NSCG occupation codes are not the SOC codes (a standard U.S. occupation list) that O\*NET uses, so they cannot be joined directly.
- The first crosswalk matched 25 occupations by hand. That covered 7,017 of the 79,800 people. A supervisor rule for two occupations removed 168 more, leaving 6,849.
- This group is not random. It leans heavily toward STEM and management jobs.
- We used O\*NET 30.2 because version 31.0 no longer has the Work Values table. In the first 25 occupations, 132 of the 150 ratings date from 2008.
- The expanded version covers 74,016 people, or 92.8% of the 79,800 eligible people. It allows occupations to share a broader profile and uses average scores for reviewed occupation groups.

## 8. Problems We Faced

| Problem | What we did or what it means |
|---|---|
| Unequal group sizes (90/10) | Used class weights, a training-only threshold, and fair metrics |
| Risk scores are inflated by class weights | Scores cannot be read as true probabilities, so we used metrics based on ranking |
| KNN limits | It cannot use class or survey weights in training and needs about 800 neighbors to be stable |
| Model ranking changes | Gradient boosting is the strongest on all 79,800 people but the weakest on the 6,849-person group |
| Small test group | The 6,849-person test set has only 124 not satisfied people, so results are uncertain |
| Linking rows back to the clean file | The clean file has no person ID, so rows are matched on 17 columns. Any change in coding could silently break this match |
| Empty `requirements.txt` | The notebook could not run from a fresh setup, so the needed packages were added |
| **One data file changed size** | `nscg2023_onet30_2_subset.csv` had 6,849 people until October 31 and now has 74,016 under the same name. Saved results show the older version, but five model notebooks read this file. Logistic regression and random forest also use `folds.csv`, which still has 6,849 rows. Re-running them may give different results or errors |

## 9. Brainstorm: Integrating Satisfaction Variables

NSCG asks how satisfied people are with nine parts of their job: advancement, benefits, intellectual challenge, independence, location, responsibility, salary, security, and contribution to society. These are the `SAT` columns. Pull request #17 added them to the expanded data. Adding them there was not a decision to use them as model inputs.

| Option | Why it could help | Risk or status |
|---|---|---|
| Keep `SAT` columns out of model inputs | Prevents data leakage | Our current rule |
| Use them to choose which work values matter most | In the EDA, people who rated an aspect very important but were dissatisfied with it had 22 to 44 points lower overall satisfaction | Analysis only, and the EDA calls this an upper bound |
| Predict "very satisfied vs. the rest" | The groups are closer in size (44/56), which gives more statistical power | Team decision needed |
| Predict all 4 levels in order | Risk scores already rise steadily across the 4 answers | Not tested yet |
| Test `SAT` columns as a separately labeled feature set | Shows how much they change a model | High leakage risk, and never part of the baseline |
| Predict one `SAT` column from the matching fit score, such as salary | Could show which job aspects fit explains best | New idea, not tested yet |

## 10. Future Steps

**November plan from the project overview:**

- **Track A:** Add fit features and measure how much they improve prediction.
- **Track B:** Compare different ways of scoring fit.
- Add a Keras neural network for comparison.

**Next technical steps:**

- Decide how to handle the changed data file in Section 8.
- Run the fit comparison on the expanded 74,016-person data.
- Use the same splits, paired bootstraps (comparing two models on the same random samples), and repeated cross-validation.
- Try KNN with a smaller set of features, such as the baseline plus one fit score.
- Tune gradient boosting before using it on small groups.

**Decisions the team still needs to make:**

1. Does `OCEDRLP` belong in the baseline or with the fit features?
2. Should the target stay "not satisfied vs. satisfied", or change to "very satisfied vs. the rest"?
3. Which threshold rule should we use? It depends on whether Swytch cares more about catching dissatisfied people or about not flagging satisfied ones.
4. What is our shared policy on survey weights in training?
5. Should we add the person ID, `OBSNUM`, to `nscg2023_clean.csv` so rows can be joined exactly?

## 11. Where to Find the Work

| Work | File |
|---|---|
| NSCG EDA and cleaning | [NSCG_2023_Exploratory_Data_Analysis.ipynb](../notebooks/NSCG_2023_Exploratory_Data_Analysis.ipynb) |
| Linking NSCG to O\*NET | [nscg_onet_data_integration.ipynb](../notebooks/possible%20data%20integration/nscg_onet_data_integration.ipynb) |
| Full evaluation | [Baseline_Model_Evaluation.ipynb](../notebooks/Baseline_Model_Evaluation.ipynb), [baseline_model_evaluation.py](../baseline_model_evaluation.py) |
| Logistic regression | [baseline_logreg.ipynb](../notebooks/baseline_logreg.ipynb) |
| Random forest | [Forest.ipynb](../notebooks/possible%20data%20integration/Forest.ipynb) |
| KNN | [KNN_Job_Satisfaction.ipynb](../notebooks/KNN_Job_Satisfaction.ipynb), [KNN_Job_Satisfaction_subset.ipynb](../notebooks/KNN_Job_Satisfaction_subset.ipynb) |
| Result tables | [baseline_results.csv](../results/baseline_results.csv), [knn_results.csv](../results/knn_results.csv), [notebooks/baseline_results.csv](../notebooks/baseline_results.csv) |
