#### OPAN 6602 - Project 1 ####

# Mike Johnson / Andrew Singh - SAXA

### Set up ----

# Libraries
library(tidyverse)
library(caret)
library(GGally)
library(broom)
library(car) # Variance inflation factor
library(readxl) # read excel files
library(pROC) #Sampling-over and under, ROC and AUC curve
library(margins) # for marginal effects

# Set random seed for reproducibility
set.seed(206)

# Set viz theme
theme_set(theme_classic())

### Load Data ----
df = read_excel("Employee_Data_Project.xlsx")



table(df$Age)


ggplot(df, aes(Age)) +
  geom_histogram()
# Data structure
str(df)

# Update data types
df = 
  df %>% 
  mutate(
    # Dependent Variable
    Attrition = factor(Attrition),
    
    # Predictors
    BusinessTravel = factor(BusinessTravel),
    Education = factor(Education, levels = 1:5, labels = c("Below College", "College", "Bachelor", "Master", "Doctor")),
    Gender = factor(Gender), 
    JobLevel = factor(JobLevel),
    MaritalStatus = factor(MaritalStatus),
    NumCompaniesWorked = as.numeric(NumCompaniesWorked),
    TotalWorkingYears = as.numeric(TotalWorkingYears), 
    EnvironmentSatisfaction = factor(EnvironmentSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High")), 
    JobSatisfaction = factor(JobSatisfaction, levels = 1:4, labels = c("Low", "Medium", "High", "Very High"))) 

# Remove Irrelevant Columns
df = 
  df %>% 
  select(
    -EmployeeID,
    -StandardHours)

# Check for NA's
na_summary = df %>% 
  summarise_all(~ sum(is.na(.))) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "na_count") %>% 
  filter(na_count > 0)

# How should we handle NAs?
na_summary

# Drop NA values
df = na.omit(df)

### Step 1: Create a train/test split ----

# Divide 30% of data to test set
test_indices = createDataPartition(1:nrow(df),
                                   times = 1,
                                   p = 0.3)

# Create training set
df_train = df[-test_indices[[1]], ]

# Create test set
df_test = df[test_indices[[1]], ]

# Create validation set
validation_indices = createDataPartition(1:nrow(df_train),
                                         times = 1,
                                         p = 0.3)

df_validation = df_train[-validation_indices[[1]],]

df_train = df_train[validation_indices[[1]],]

### Step 2: Data Exploration ----

# Summary of training set
summary(df_train)

#df_train %>% 
# ggpairs(aes(color = Attrition, alpha = 0.4))

# Viz of attrition distribution
# Imbalanced classes. Will need to downsample.
df_train %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Gender, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Gender Distribution by Attrition")

# Viz of relationship between Age and Attrition
df_train %>% 
  ggplot(aes(x = Age, fill = Attrition)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Age Distribution by Attrition")

df_train %>% 
  mutate(age_t = log(Age)) %>% 
  ggplot(aes(x = age_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Age Transformed Distribution by Attrition")
  

# Viz of relationship between Education and Attrition
df_train %>% 
  ggplot(aes(x = Education, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Education Distribution by Attrition")

# Viz of relationship between Job Satisfaction and Attrition
df_train %>% 
  ggplot(aes(x = JobSatisfaction, fill = Attrition)) +
  geom_bar() +
  facet_grid(~Attrition) +
  labs(title = "Job Satisfaction Distribution by Attrition")

# Viz of relationship between Working Years and Attrition
df_train %>% 
  ggplot(aes(x = TotalWorkingYears, fill = Attrition)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  facet_grid(~Attrition) +
  labs(title = "Working Years Distribution by Attrition")

df_train %>% 
  mutate(workingyears_t = log(TotalWorkingYears)) %>% 
  ggplot(aes(x = workingyears_t, fill = Attrition)) +
  geom_histogram() +
  facet_grid(~Attrition) +
  labs(title = "Working Years Transformed Distribution by Attrition")
  

### Step 3: Data pre-processing ----

# Downsampling
downsample_df = downSample(x = df_train[ , colnames(df_train) != "Attrition"],
                           y = df_train$Attrition)

colnames(downsample_df)[ncol(downsample_df)] = "Attrition"

downsample_df %>% 
  ggplot(aes(x = Attrition)) + 
  geom_bar(fill = "steelblue") +
  labs(title = "Attrition Distribution")

### Step 4: Feature Engineering ----


### Step 5: Feature & Model Selection ----

# Initial Model
f1 = glm(
  Attrition ~ . +
    I(Age ^ 2) +
    I(DistanceFromHome ^ 2) +
    I(Income ^ 2) +
    I(NumCompaniesWorked ^ 2) +
    I(TotalWorkingYears ^ 2) +
    I(TrainingTimesLastYear ^ 2) +
    I(YearsAtCompany ^ 2) +
    I(YearsWithCurrManager ^ 2),
  data = downsample_df,
  family = binomial("logit"))

summary(f1)

vif(f1)

roc1 = roc(
  data = 
    tibble(
      actual = 
        df_train %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
        predicted = predict(f1, df_train)),
  "actual",
  "predicted"
)

plot(roc1)

roc1$auc

# Stepwise Regression
f_step = step(object = f1,
              direction = "both")

summary(f_step)

vif(f_step)

roc_step = roc(
  data = 
    tibble(
      actual = 
        df_train %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = predict(f_step, df_train)),
  "actual",
  "predicted"
)

plot(roc_step)

roc_step$auc
roc1$auc

# Final Model
f_final = glm(
  Attrition ~
    Age +
    BusinessTravel +
    MaritalStatus +
    NumCompaniesWorked +
    JobSatisfaction +
    TotalWorkingYears + 
    I(YearsAtCompany^2),
  data = downsample_df,
  family = binomial("logit"))

summary(f_final)

vif(f_final)

roc_final = roc(
  data = 
    tibble(
      actual = 
        df_train %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = predict(f_final, df_train)),
  "actual",
  "predicted"
)

plot(roc_final)

roc_final$auc
roc_step$auc
roc1$auc

### Step 6: Model Validation ----

preds_validation = predict(f_final, df_validation)

roc_validation = roc(
  data = 
    tibble(
      actual = 
        df_validation %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = preds_validation),
  "actual",
  "predicted"
)

plot(roc_validation)

roc_final$auc
roc_validation$auc

### Step 7: Predictions and Conclusions ----

preds_test = predict(f_final, df_test)

roc_test = roc(
  data = 
    tibble(
      actual = 
        df_test %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = preds_test),
  "actual",
  "predicted"
)

plot(roc_final)

roc_final$auc
roc_test$auc

# Re-train the model on the whole data set for marginal effects/production

# Downsampling
downsample_prod = downSample(x = df[ , colnames(df) != "Attrition"],
                           y = df$Attrition)

colnames(downsample_prod)[ncol(downsample_prod)] = "Attrition"

# Production Model
f_prod = glm(
  Attrition ~
    Age +
    BusinessTravel +
    MaritalStatus +
    NumCompaniesWorked +
    JobSatisfaction +
    TotalWorkingYears + 
    I(YearsAtCompany^2),
  data = downsample_prod,
  family = binomial("logit"))

summary(f_prod)

roc_prod = roc(
  data = 
    tibble(
      actual = 
        df %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = predict(f_prod, df)),
  "actual",
  "predicted"
)

plot(roc_prod)

roc_prod$auc

# Marginal Effects
coefs = 
  tidy(f_prod) %>% 
  mutate(odds = exp(estimate),
         odds_mfx = odds - 1)

coefs

mfx = margins(f_prod)

summary(mfx)

summary(f_prod)


### Model Comparison ----

# Create Models
model_1 = glm(Attrition ~ Age,
              data = downsample_df,
              family = binomial("logit"))

model_2 = glm(Attrition ~ Age + Gender,
              data = downsample_df,
              family = binomial("logit"))

model_3 = glm(Attrition ~ Age + Gender + JobSatisfaction,
              data = downsample_df,
              family = binomial("logit"))

model_4 = glm(Attrition ~ Age + Gender + JobSatisfaction + Income + Gender:Income,
              data = downsample_df,
              family = binomial("logit"))

##  Validate models

# Function to calculate model metrics

calc_metrics = function(model, data) {
  predicted_prob = predict(model, data, type = "response")
  predicted_class = ifelse(predicted_prob > 0.5, "Yes", "No")
  actual_values = data$Attrition
  
  auc = auc(roc(actual_values, predicted_prob))
  conf_matrix = confusionMatrix(factor(predicted_class, 
                                       levels = c("No", "Yes")),
                                factor(actual_values, levels = c("No", "Yes")))
  precision = conf_matrix$byClass['Precision']
  recall = conf_matrix$byClass['Recall']
  
  return(list(AIC = AIC(model), AUC = auc, Precision = precision, Recall = recall))
}


# Create a table with all values
calc_metrics(model_1, df_validation)
calc_metrics(model_2, df_validation)
calc_metrics(model_3, df_validation)
calc_metrics(model_4, df_validation)

# Create a graph with all values
roc_data <- bind_rows(
  mutate(roc_1_validation, Model = "Model 1"),
  mutate(roc_2_validation, Model = "Model 2"),
  mutate(roc_3_validation, Model = "Model 3"),
  mutate(roc_4_validation, Model = "Model 4")
)

# Plot the ROC curves
ggplot(roc_data, aes(x = 1 - Specificity, y = Sensitivity, color = Model)) +
  geom_line(linewidth = 1) +
  labs(
    title = "ROC Curve Comparison for Logistic Regression Models",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")


# Actual Values
actual_values = 
  df_validation %>% 
  select(Attrition) %>% 
  unlist()

# Model 1
model_1_validation = predict(model_1, df_validation)

roc_1_validation = roc(
  data = 
    tibble(
      actual = 
        df_validation %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = model_1_validation),
  "actual",
  "predicted"
)

plot(roc_1_validation)

roc_1_validation$auc

# Model 2
model_2_validation = predict(model_2, df_validation)

roc_2_validation = roc(
  data = 
    tibble(
      actual = 
        df_validation %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = model_2_validation),
  "actual",
  "predicted"
)

plot(roc_2_validation)

roc_2_validation$auc

# Model 3
model_3_validation = predict(model_3, df_validation)

roc_3_validation = roc(
  data = 
    tibble(
      actual = 
        df_validation %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = model_3_validation),
  "actual",
  "predicted"
)

plot(roc_3_validation)

roc_3_validation$auc

# Model 4
model_4_validation = predict(model_4, df_validation)

roc_4_validation = roc(
  data = 
    tibble(
      actual = 
        df_validation %>%  # not using balanced data for evaluation
        select(Attrition) %>% 
        unlist(),
      predicted = model_4_validation),
  "actual",
  "predicted"
)

plot(roc_4_validation)

roc_4_validation$auc





## Part 3.1

# Calculate the marginal distribution of Attrition
marginal_distribution <- prop.table(table(df$Attrition))

# Print the results
cat("Estimated Marginal Distribution of Attrition:\n")
print(marginal_distribution)

## Part 3.2


# Define the younger and older age groups
younger_age <- 25
older_age <- 55

# Filter the data for the specified age groups
younger_group <- subset(df, Age == younger_age)
older_group <- subset(df, Age == older_age)

# Calculate the attrition rate for each group
younger_attrition_rate <- sum(younger_group$Attrition == "Yes") / nrow(younger_group)
older_attrition_rate <- sum(older_group$Attrition == "Yes") / nrow(older_group)

# Replace NaN with 0 if there are no records for a group
younger_attrition_rate <- ifelse(is.nan(younger_attrition_rate), 0, younger_attrition_rate)
older_attrition_rate <- ifelse(is.nan(older_attrition_rate), 0, older_attrition_rate)

# Print the results
cat("Attrition rate for younger employees (age 25):", younger_attrition_rate, "\n")
cat("Attrition rate for older employees (age 55):", older_attrition_rate, "\n")


## 4.1

# Convert Attrition to a binary variable for logistic regression
df$AttritionBinary <- ifelse(df$Attrition == "Yes", 1, 0)

# Fit a logistic regression model
logit_model <- glm(AttritionBinary ~ Age, data = df, family = binomial)

# Create a sequence of ages for prediction
age_range <- seq(min(df$Age), max(df$Age), length.out = 100)

# Predict probabilities of attrition using the logistic model
predicted_probs <- predict(logit_model, newdata = data.frame(Age = age_range), type = "response")

summary(predicted_probs)

# Plot the relationship
plot(df$Age, df$AttritionBinary,
     xlab = "Age",
     ylab = "Probability of Attrition",
     main = "Relationship Between Age and Attrition",
     pch = 19, col = "blue",
     xlim = c(min(age_range), max(age_range)),
     ylim = c(0, 1))

# Add the logistic regression curve
lines(age_range, predicted_probs, col = "red", lwd = 2)

# Add legend
legend("topright", legend = c("Observed Data", "Logistic Curve"), 
       col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 1), lwd = c(NA, 2))



# Load necessary library for Excel export (optional)
# install.packages("writexl") # Uncomment if not installed
library(writexl)

# Create the data frame (your table data)
mean_sales <- data.frame(
  term = c("(Intercept)", "Age", "BusinessTravelTravel", "BusinessTravelTravel",
           "MaritalStatusMarried", "MaritalStatusSingle", "NumCompaniesWorked", 
           "JobSatisfactionMedium", "JobSatisfactionHigh", "JobSatisfactionVeryHigh", 
           "TotalWorkingYears", "I(YearsAtCompany^2)"),
  estimate = c(0.0819, -0.0243, 1.61, 0.949, 0.191, 0.972, 0.176, -0.492, 
               -0.443, -0.883, -0.0943, 0.00271),
  std.error = c(0.360, 0.00872, 0.251, 0.227, 0.168, 0.171, 0.0263, 0.181, 
                0.162, 0.170, 0.0148, 0.000519),
  statistic = c(0.227, -2.78, 6.41, 4.19, 1.14, 5.67, 6.68, -2.71, -2.73, 
                -5.20, -6.35, 5.22),
  p.value = c(8.20e-1, 5.37e-3, 1.41e-10, 2.82e-5, 2.54e-1, 1.41e-8, 2.47e-11, 
              6.68e-3, 6.29e-3, 2.00e-7, 2.11e-10, 1.78e-7),
  odds = c(1.09, 0.976, 5.00, 2.58, 1.21, 2.64, 1.19, 0.612, 0.642, 
           0.414, 0.910, 1.00),
  odds_mfx = c(0.0853, -0.0240, 4.00, 1.58, 0.211, 1.64, 0.192, -0.388, 
               -0.358, -0.586, -0.0900, 0.00271)
)

# View the data frame to verify its structure
print(mean_sales)

# Export the data to a CSV file
write.csv(mean_sales, "exported_table.csv", row.names = FALSE)

# Export the data to an Excel file (if you prefer Excel format)
write_xlsx(mean_sales, "exported_table.xlsx")







