#R Analysis performed by Hendrik Erenstein, June 11th 2025

echo=TRUE
library(parameters)
library(lme4)
library(merDeriv)
data <- read.table('MAIN PATH.csv', sep=",", header=T)

#List columns
colnames(data)

#Exclude specific observers due to expertise
data <- data[!data$Observer %in% c('I', 'F', 'A'), ]
#Drop HD NA values
data <- data[!is.na(data$Hausdorff_mm), ]

muscles <- unique(data$Muscle)
models <- list()

for (muscle in muscles) {
formula <- as.formula(paste("Hausdorff_mm ~ Experience.binary * (1 | Observer) + (1 | Volume)"))
data_subset <- subset(data, Muscle == muscle)
models[[muscle]] <- lmer(formula, data = data_subset)
}

for (muscle in muscles) {
  print('***************************')
  print('***************************')
  print(paste("Results for muscle:", muscle))
  #Print T and p-values
  print(paste('Coefficient: ', model_parameters(models[[muscle]])$Coefficient[2]))
  print(paste('T-value: ', model_parameters(models[[muscle]])$t[2]))
  print(paste('p-value: ', model_parameters(models[[muscle]])$p[2]))
  print('***************************')
}

#Translate to dataframe
results_df <- data.frame(
  Muscle = character(),
  Coefficient = numeric(),
  T_value = numeric(),
  P_value = numeric(),
  stringsAsFactors = FALSE
)
for (muscle in muscles) {
  results_df <- rbind(results_df, data.frame(
    Muscle = muscle,
    Coefficient = model_parameters(models[[muscle]])$Coefficient[2],
    T_value = model_parameters(models[[muscle]])$t[2],
    P_value = model_parameters(models[[muscle]])$p[2]
  ))
}
#Round the dataframe to 3 decimal places except Muscle column
results_df$Coefficient <- round(results_df$Coefficient, 3)
results_df$T_value <- round(results_df$T_value, 3)
results_df$P_value <- round(results_df$P_value, 3)

# Print the results dataframe
print(results_df)


library(influence.ME)
# Influence analysis
influence_results <- list()
for (muscle in muscles) {
  data_subset <- subset(data, Muscle == muscle)
  model <- models[[muscle]]
  influence_results[[muscle]] <- max(cooks.distance(influence(model, group="Observer")))
}
influence_results


# Compute influence measures
infl <- influence(model, group = "Observer")  # or "Volume", depending on your grouping

# Cook's distance
cooks <- cooks.distance(infl)
print(cooks)



#####
#robust
library(robustlmm)
results_df_robust <- data.frame(
  Muscle = character(),
  Coefficient = numeric(),
  T_value = numeric(),
  P_value = numeric(),
  stringsAsFactors = FALSE
)
for (muscle in muscles) {
  data_subset <- subset(data, Muscle == muscle)
  print('***************************')
  print('***************************')
  print(paste("Results for muscle:", muscle))
  #Print T and p-values
  robust_model <- rlmer(formula, data = data_subset)
  print(paste('r Coefficient: ', model_parameters(robust_model)$Coefficient[2]))
  print(paste('r T-value: ', model_parameters(robust_model)$t[2]))
  print(paste('r p-value: ', model_parameters(robust_model)$p[2]))
  print('***************************')
  #Add to results and round to 3 decimal places
  results_df_robust <- rbind(results_df_robust, data.frame(
    Muscle = muscle,
    Coefficient = round(model_parameters(robust_model)$Coefficient[2], 3),
    T_value = round(model_parameters(robust_model)$t[2], 3),
    P_value = round(model_parameters(robust_model)$p[2], 3)
  ))
}
# Print the robust results dataframe
print(results_df_robust)
