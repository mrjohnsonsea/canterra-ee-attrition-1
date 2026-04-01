# Canterra Employee Attrition — Predictive Modeling

**Georgetown University MSBA | OPAN 6602: Machine Learning I**
**Authors:** Andrew Singh & Mike Johnson

---

## Business Problem

Canterra faces a **15% annual attrition rate**, resulting in approximately 600 employees leaving each year. Leadership identified three primary organizational costs: disruption to ongoing project timelines, degraded capacity requiring constant backfill, and ongoing training burden for incoming talent.

The analysis addresses two stakeholder questions:

1. **Leadership hypothesis** — Do higher job satisfaction and greater total working years reduce attrition?
2. **Marketing department inquiry** — How do demographic factors (age, gender, education) relate to employee attrition?

---

## Approach

**Model type:** Logistic regression — appropriate for a binary outcome (attrition: Yes / No) and chosen for its interpretability and ability to control for multicollinearity across predictors.

**Data:** `Employee_Data_Project.xlsx` — employee records including demographic, job satisfaction, and career history variables.

**Pre-processing:**
- Recoded six variables as categorical factors: Business Travel, Education, Gender, Job Level, Marital Status, and Satisfaction (Job and Environment)
- Converted Number of Companies Worked and Total Working Years to numeric
- Removed non-predictive identifiers (Employee ID, Standard Hours)
- Dropped records with missing values (minimal NAs across a small number of variables)

**Train / Validation / Test split:**
- 30% held out as test set
- Of the remaining 70%, a further 30% held out as validation set
- Training set used for model fitting; validation for model selection; test set for final evaluation

**Class imbalance:** The training data showed an imbalanced attrition distribution. Downsampling was applied to the training set to produce balanced classes before model fitting.

---

## Model Selection

A broader feature selection process — starting from a full model with quadratic terms and applying stepwise regression — produced the **final predictive model**:

```
Attrition ~ Age + BusinessTravel + MaritalStatus + NumCompaniesWorked
           + JobSatisfaction + TotalWorkingYears + I(YearsAtCompany²)
```

---

## Final Model Performance

| Metric | Value |
|--------|-------|
| AUC | 0.74 |
| AIC | 355.09 |
| Precision | 0.92 |
| Recall | 0.65 |

The model achieves a good level of predictive accuracy for a binary attrition classifier.

---

## Key Findings

### Average Marginal Effects on Attrition Probability

Statistically significant predictors (p < 0.05):

| Variable | Avg. Marginal Effect | Direction |
|----------|---------------------|-----------|
| Business Travel: Frequently | +0.339 | Increases attrition |
| Business Travel: Rarely | +0.207 | Increases attrition |
| Marital Status: Single | +0.217 | Increases attrition |
| Number of Companies Worked | +0.031 | Increases attrition |
| Years at Company (Squared) | +0.007 | Increases attrition |
| Job Satisfaction: Very High | -0.197 | Decreases attrition |
| Job Satisfaction: High | -0.121 | Decreases attrition |
| Job Satisfaction: Medium | -0.098 | Decreases attrition |
| Total Working Years | -0.022 | Decreases attrition |

*Note: Age and Marital Status: Married were not statistically significant at p < 0.05.*

### Exploratory Findings

- **Marginal attrition rate:** 16% of employees experienced attrition (84% did not)
- **Age and attrition:** Younger employees (age 25) had a 23% attrition rate vs. ~14% for older employees (age 55). As age increases, probability of attrition decreases
- **Gender:** Males outnumbered females in both attrition and retention categories; gender was not identified as a significant predictor of attrition
- **Education:** Employees with bachelor's or master's degrees were more likely to experience attrition than those without higher education; doctoral-level employees were least likely to leave

---

## Recommendations

Based on the model findings, the following actions are recommended for Canterra:

1. **Support employees who travel** — Provide improved travel allowances, wellness programs, and technology alternatives (e.g., video conferencing) to reduce the burden and attrition risk associated with frequent travel.

2. **Engage single employees** — Develop targeted engagement strategies (social events, mentorship programs, community-building initiatives) to strengthen organizational connection among single employees.

3. **Invest in job satisfaction** — Conduct regular employee satisfaction surveys and invest in recognition programs, career development opportunities, and work-life balance improvements.

4. **Incentivize long-term commitment** — Offer clear career progression paths, tailored professional development plans, and retention bonuses to reward tenure.

---

## Repository Contents

| File | Description |
|------|-------------|
| `analysis.R` | Full R analysis: data loading, EDA, preprocessing, model selection, validation, and marginal effects |
| `Employee_Data_Project.xlsx` | Employee dataset used for the analysis |
| `Machine Learning I Project 2.pdf` | Written report with findings, recommendations, and technical appendix |

---

## Tools & Libraries

**Language:** R

**Libraries:** `tidyverse`, `caret`, `broom`, `car`, `readxl`, `pROC`, `margins`, `GGally`
