install.packages("MASS")
install.packages("DMwR")
install.packages("lattice")
install.packages("grid")
install.packages("caTools")
library(plyr)
library(dplyr)
library(MASS)
library(lattice)
library(grid)
library(DMwR)
library(caTools)
library(ggplot2)

set.seed(88)
sephora <- read.csv(file.choose(), header = TRUE, sep = ",")

# Drop rows with missing values in size_cleaned_oz column and rating is equal to zero. In total, 3580 rows have been dropped.
sephora_cleaned <- filter(sephora, size_cleaned_oz != 0)
sephora_cleaned <- filter(sephora_cleaned, rating != 0)

#Summary of sephora rating
summary(sephora_cleaned$rating)
str(sephora_cleaned)

plot_sephora_rating <- ggplot(sephora_cleaned_subsetted, aes(x = rating)) + geom_bar(fill = "blue",  color = "white", alpha = 0.3) + 
  ylab("Frequency") +
  xlab("Rating") 
plot_sephora_rating

#Convert rating to ordinal variable
sephora_cleaned$rating <- ordered(sephora_cleaned$rating)

#Create dummy variables for how_to_use, ingredients, and options
sephora_cleaned$how_to_use_featured <- ifelse(sephora_cleaned$how_to_use == "no instructions", 0, 1)
sephora_cleaned$ingredients_featured <- ifelse(sephora_cleaned$ingredients == "unknown", 0, 1)
sephora_cleaned$options_featured <- ifelse(sephora_cleaned$options == "no options", 0, 1)

# Remove columns that that are of characters type 
sephora_cleaned_subsetted <- subset(sephora_cleaned, select = -c(brand, category, name, size, size_cleaning, URL, MarketingFlags, MarketingFlags_content, 
                                                                 options, details, how_to_use, ingredients))
sephora_cleaned_subsetted$category_New <- as.factor(sephora_cleaned_subsetted$category_New)

#Price has a left skewed distribution
plot_value_price <- ggplot(sephora_cleaned_subsetted, aes(x = value_price)) + geom_histogram(fill = "blue",  color = "white", alpha = 0.3, bins = 20) + 
  ylab("Frequency") +
  xlab("Value Price") 
plot_value_price

plot_value_price_log <- ggplot(sephora_cleaned_subsetted, aes(x = log(value_price))) + geom_histogram(fill = "blue",  color = "white", alpha = 0.3, bins = 20) + 
  ylab("Frequency") +
  xlab("Log Transformed Value Price") 
plot_value_price_log

#Sample split into training and prediction by 90/10
split <- sample.split(sephora_cleaned_subsetted$rating, SplitRatio = 0.8)

sephora_cleaned_training <- subset(sephora_cleaned_subsetted, split == TRUE)
sephora_cleaned_prediction <- subset(sephora_cleaned_subsetted, split == FALSE)
summary(sephora_cleaned_training$rating)
summary(sephora_cleaned_prediction$rating)

# SMOTE
table(sephora_cleaned_training$rating)
str(sephora_cleaned_training$rating)

sephora_cleaned_balanced <- SMOTE(rating ~., data = sephora_cleaned_training, perc.over = 3000, perc.under = 800, k = 3)
summary(sephora_cleaned_balanced$rating)
#

#Adapted from https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/:
#Ordinal logistic regression
fit <- polr(rating ~ Sephora_brand + category_New + size_cleaned_oz + log(value_price) + online_only + exclusive + 
              limited_edition + limited_time_offer + how_to_use_featured + ingredients_featured + options_featured, 
            data = sephora_cleaned_balanced, Hess = TRUE)

summary(fit)

#Carry out stepwise regression to remove insignificant variables based on infomration criteria (AIC)
fit_stepwise <- stepAIC(fit, direction = c("both"), trace = 1)

summary(fit_stepwise)

#Calculate p value by comparing the t-value against the standard normal distribution, like a z test. 
#Howeover, this is only true within infinite degrees of freedom, but is reasonably approximated by large samples.
coefficient_table <- coef(summary(fit_stepwise))
p_value <- pnorm(abs(coefficient_table[,"t value"]), lower.tail = FALSE) * 2
coefficient_table <- cbind(coefficient_table, "p value" = round(p_value, 3))
coefficient_table

#Calculate confidence intervals for the parameter estimates.
ci <- confint(fit) # default method gives profiled CIs
ci

#Predict rating for the prediction data set
summary(sephora_cleaned_prediction$rating)

predicted.rating <- predict(fit_stepwise, sephora_cleaned_prediction)

summary(predicted.rating)

summary(sephora_cleaned_prediction$rating)

#New Product Rating
#Example: Sephora Cream Lip Stain Liquid Lipstick
#https://www.sephora.com/ca/en/product/cream-lip-stain-liquid-lipstick-P281411?icid2=products%20grid:p281411
Sephora_Palette <- data.frame("category_New" = "Makeup", "size_cleaned_oz" = 0.169, "value_price" = log(19), "exclusive" = 1, "ingredients_featured" = 1) 
round(predict(fit_stepwise, Sephora_Palette, type = "p"), 3) * 100

#Example: Be Gentle, Be Kind Aloe + Oat Milk Ultra Soothing Fragrance-Free Shampoo
#https://www.sephora.com/ca/en/product/briogeo-be-gentle-be-kind-aloe-oat-milk-ultra-soothing-fragrance-free-shampoo-P461428?icid2=justarrivedhair_skugrid_ufe:p461428:product
briogeo <- data.frame("category_New" = "Hair", "size_cleaned_oz" = 8, "value_price" = log(36), "exclusive" = 0, "ingredients_featured" = 1) 
round(predict(fit_stepwise, briogeo, type = "p"), 3) * 100

#Export Rating to generate classfication report in Python
write.csv(sephora_cleaned_prediction$rating, file = "Actual Rating.csv")
write.csv(predicted.rating, file = "Predicted Rating.csv")
