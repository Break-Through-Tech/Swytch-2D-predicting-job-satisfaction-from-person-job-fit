# Predicting Job Satisfaction from Person-Job Fit

> 💡In this project, you will use national survey data on job satisfaction (e.g., the General Social Survey) together with the O*NET database of what occupations involve, and machine learning methods (feature engineering, logistic regression, tree-based models, k-nearest neighbors, and a Keras neural network), to predict whether someone is satisfied at work based on how well their personal work values match what their job actually offers. This tests a core idea behind SWYTCH: that matching people on what they value, not just on job titles, leads to better outcomes.

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Jenna Hunte      | @jehunte      | Break Through Tech Coach                                                 |
|                  | @             |                                                                          |
|                  | @             |                                                                          |
|                  | @             |                                                                          |
|                  | @             |                                                                          |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

## 🧪 Baseline Model Evaluation

The October milestone focuses on a fair baseline comparator that uses only background factors instead of the later person-job fit features. The evaluation script at [baseline_model_evaluation.py](baseline_model_evaluation.py) implements this setup and is designed to keep the crosswalk and occupation-linking work separate from the prediction task itself.

The baseline workflow uses:

- Stratified train/test splits so the satisfied and unsatisfied groups are represented fairly even when they are imbalanced.
- Metrics that are not fooled by a majority-class baseline: ROC AUC, average precision, balanced accuracy, precision, recall, and F1.
- A short list of qualitative properties for the model review:
  1. Predictive accuracy: does the model separate satisfied and unsatisfied workers?
  2. Satisfaction signal: does it meaningfully capture job-satisfaction variation without simply predicting the majority class?
  3. Group sensitivity: does performance remain stable across demographic or work-status groups?
- Models tried in the evaluation: majority-class baseline, logistic regression, random forest, and a gradient-boosted tree model.

This keeps the baseline honest while preserving the later comparison against fit-based features. The script also includes a group-level sensitivity report to check whether satisfaction predictions vary sharply by sex or similar categories.

The script runs on the cleaned NSCG 2023 file (`data/college_grads_2023/nscg2023_clean.csv`, 79,800 employed college graduates). The target is `not_satisfied` (somewhat or very dissatisfied, 9.9%). It uses the same features, split and model settings as [notebooks/Baseline_Model_Evaluation.ipynb](notebooks/Baseline_Model_Evaluation.ipynb), so its output matches `results/baseline_results.csv`. Run it with `python baseline_model_evaluation.py` (about 15 seconds).

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
