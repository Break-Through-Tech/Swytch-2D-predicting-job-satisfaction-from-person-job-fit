# Predicting Job Satisfaction from Person-Job Fit

**Company / Org:** Swytch  
**Challenge Advisor:** James Thompson, [jimt@swytch.careers]  
**AI Studio Coach:** Jenna Hunte, jenna.hunte@breakthroughtech.org     
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Swytch
Swytch is a career-tech organization dedicated to optimizing professional alignment by focusing on core work values rather than traditional job titles. Their team aims to leverage data science to improve job satisfaction outcomes by effectively matching individuals with roles that provide what they truly need.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use national survey data on job satisfaction (e.g., the General Social Survey) together with the O*NET database of what occupations involve, and machine learning methods (feature engineering, logistic regression, tree-based models, k-nearest neighbors, and a Keras neural network), to predict whether someone is satisfied at work based on how well their personal work values match what their job actually offers. This tests a core idea behind SWYTCH: that matching people on what they value, not just on job titles, leads to better outcomes.

### Success Criteria
A working pipeline that links the survey data to occupational profiles. Models that predict job satisfaction, with a clear read on how much the fit features improve prediction over background factors alone (reported with standard measures like AUC, plus confidence intervals). A comparison of fit-scoring methods showing which ones actually help. And an honest writeup of how big the effect is and where the limits are. The scoring recipe stays in-house; students report findings only.

### Stretch Goals

- Free option: break the results out by industry or demographic group to see where the value-satisfaction link is strongest and weakest, or test how stable the findings are across survey years.
- Optional API-based add-on (TBD, only if API access is later funded): have the model generate plain-language explanations of a person's fit, or a question-answering assistant over the survey codebook and O*NET docs.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Data Linkage & Exploration | Match the survey's occupation codes to O*NET. Combine each person's work values and job satisfaction with their occupation's profile. Explore the data. |
| October | Baseline Models & Evaluation Setup | Build simple baseline models using only background factors (income, hours, demographics). Set up a fair way to measure results when the satisfied and unsatisfied groups are different sizes. |
| November | Fit Features & Model Comparison | Add the "fit" features. Track A measures how much they improve the prediction; Track B compares different ways of scoring fit. Add a Keras neural network for comparison. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Suggested Datasets
**Name and Source:** General Social Survey (GSS)
**Format:** SPSS (can read with pyreadstat library)
**Size:** under 1gb  
**Location:** https://gss.norc.org/get-the-data.html

**Name and Source:  O*NET Database  
**Format:** Excel, CSV, Json, or SQL 
**Size:** under 1gb  
**Location:** https://www.onetcenter.org/crosswalks.html

**Name and Source:** The American Job Quality Study: 2025  
**Format:** CSV / TSV  
**Size:** under 1gb  
**Location:** https://www.gallup.com/analytics/691241/american-job-quality-study.aspx

**Name and Source:** National Survey of College Graduates (NSCG): 2023  
**Format:** Excel, CSV, or SAS (can read with pyreadstat library)
**Size:** under 1gb  
**Location:** https://ncses.nsf.gov/surveys/national-survey-college-graduates/2023

### Key Details
- The core analysis combines **individual-level survey responses from the GSS** with **occupation-level information from O*NET**. The team will need to use occupation codes and the appropriate O*NET crosswalk to connect the two.
- The modeling dataset should ultimately contain, for each individual, their **job satisfaction, personal work values, occupation, occupation-level characteristics, and selected background factors**.
- A central challenge is defining and measuring **person-job fit**: how closely an individual's reported work values align with the characteristics or values associated with their occupation.
- The team should investigate **missing values, survey-year differences, occupation-code changes, sample sizes, and other data-quality issues** before building models.
- Clearly document all decisions about **filtering, transformations, feature engineering, and fit-score construction** so that the modeling results can be reproduced.
- The project should distinguish between **features available for the baseline model** and the additional **person-job fit features**. This distinction is important for measuring whether fit actually improves prediction.
- Review the documentation and codebooks for each dataset before beginning analysis. Do not assume that similarly named variables across datasets measure the same thing.
  
---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Regression / Logistic Regression, Deep Learning / Neural Networks

The primary goal is to predict whether an individual is satisfied at work. The project should compare models using two sets of features:

1. **Baseline features:** Background factors such as income, hours worked, and demographic characteristics.
2. **Person-job fit features:** Features that capture how closely an individual's work values align with characteristics of their occupation.

The key question is not simply which model performs best, but **whether adding person-job fit information meaningfully improves prediction over the baseline.**

### Suggested Modeling Approaches

- Logistic regression
- Tree-based models
- K-nearest neighbors
- Keras neural network

You do not necessarily need to use every approach if your analysis shows that some models are not appropriate. Explain and justify your modeling choices.

**Recommended Libraries:**
- pandas
- NumPy
- scikit-learn
- pyreadstat
- TensorFlow / Keras
- Matplotlib / Seaborn

**Evaluation Metrics:**
- **Precision, recall, and F1:** Useful for understanding classification performance at a selected threshold.
- **Confusion matrix:** Useful for examining the types of errors models make.
- **Confidence intervals:** Report uncertainty around key performance estimates, particularly AUC.
- **Baseline vs. fit-feature comparison:** Quantify how much predictive performance changes when person-job fit features are added.

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- **Person-Job Fit:** Read about the concept of person-job fit and how alignment between an individual's characteristics, values, and work environment may relate to job satisfaction.
- **Job Satisfaction:** Review research on the factors associated with job satisfaction and how work values and job characteristics can influence satisfaction.
- **Machine Learning for Tabular Data:** Review the strengths and limitations of common approaches for structured/tabular data, including logistic regression, tree-based models, K-nearest neighbors, and neural networks.
- **Model Evaluation:** Review how classification models are evaluated, with particular attention to ROC AUC, precision/recall, class imbalance, cross-validation, and confidence intervals.

**Technical Tutorials & Documentation:**
- [scikit-learn classification documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn model evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Keras documentation](https://keras.io/)
- [pyreadstat documentation](https://pyreadstat.readthedocs.io/)

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)
I will give unsolicited suggested guidance during these meetings. I am open to any and all questions.

 **Other ways to reach out to me with questions:** 
* jimt@swytch.careers,  jenna.hunte@breakthroughtech.org
* Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting.
2. **Review the GSS, O*NET, and American Job Quality Study documentation** to understand what each dataset contains before beginning analysis.
3. **Begin exploring the datasets**, paying particular attention to occupation codes, work-value variables, job-satisfaction variables, missing data, and potential linkage challenges.
4. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
