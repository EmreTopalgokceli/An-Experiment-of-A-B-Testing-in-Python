##########################################
## An Experiment of A/B Testing in Python
#########################################

# A restaurant manager is considering converting the non-smoking area of her restaurant into a smoking-allowed area.
# She looked at the average of the total amount paid at smoking and non-smoking tables and found the results that smokers
# pay more.

# Should she trust this result and make her decision accordingly? Or with a more obvious question, could the result have
# happened by chance?

# We made an experiment of statistical hypothesis testing (also known as A/B Testing) in Python to have a proper answer
# to this question.

# In our experiment, we are simply aiming to compare the mean of two groups (average bills paid by smokers and non-smokers)
# and see if the existing difference between them is statistically significant.

# Let’s start by importing the libraries we will use during our experiment, keeping in mind that the restaurant manager
# is sitting next to us and looking forward to the result we will give her.

import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr
from statsmodels.stats.proportion import proportions_ztest

# We will use the dataset named tips through the Python package Seaborn.

import seaborn as sns
tips = sns.load_dataset("tips")

tips.shape
tips.info()

# We can clearly see at the below prints that our data has 7 variables, 2 of which are numeric, and 244 observations for
# each variable. There is no missing variable. We have numerical, categorical, and time variables.

sns.catplot(x="day", y="total_bill", hue="smoker", kind="violin", split=True, data=tips)

# total_bill is the cost of the meal eaten per party. Tax is included.
# tip is tip (gratuity) amount per party.
# sex is the sex of the person paying for the meal (0=male, 1=female)
# smoker refers to whether the table is in a smoke-allowed area or not. (0=no, 1=yes)
# day is the day on which meal is eaten (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time is the time of meal (0=lunch, 1=dinner)
# size is the size of the party
# All figures are in US dollars.

# After having a sense of the data, we can start by taking the groups’ averages using the method of the groupby in Pandas
# library. At first glance, the data gives us the idea that smokers pay in average USD 1.57 more than non-smokers do.

tips.groupby("smoker").agg({"total_bill": "mean"})

#        total_bill
# smoker
# Yes      20.756344
# No       19.188278

# Based on the above figures, the manager is of the opinion that turning her restaurant completely into a smoke-allowed
# place is more profitable.

# But what if this difference occurred by chance? Could there be another factor affecting the mean of the groups rather
# than the smoking? We will employ A/B Testing here to answer these questions.

#########################
## Designing A/B Testing
#########################

# First, we need to establish our hypothesis, and then we will check whether our data meet the assumption of normality
# and homogeneity of variance to see if we should use parametric or non-parametric hypothesis testing. The final test
# we will employ to check the below hypothesis will be decided based on whether the normality and the variance homogeneity
# assumptions be met.

# Let’s set our hypothesis up first.

# Ho: μ0 — μ1 = 0
# Ha: μ0 — μ1 ≠ 0

# Which also means,
# Null Hypothesis (NH or Ho): Assumes no difference between the variables. (In our experiment, it refers that there is
# no statistically significant difference between the amount of the total bill paid by smokers and non-smokers.)
# Alternative Hypothesis (AH or Ha): Assumes a difference between the variables. (In our experiment, it refers that there
# is no statistically significant difference between the amount of the total bill paid by smokers and non-smokers.)

# We will also use the p-value to evaluate the result of all hypotheses be employed in our experiment. P-value is the
# probability (or chance) of getting the collected data under the assumption of the null hypothesis. If this probability
# is smaller 0.05, ‘Ho is rejected in favour of Ho’, termed a ‘statistically significant result’; otherwise ‘fail
# to reject Ho’, termed a ‘non-statistically significant result’.

##############
# Assumption of normality
###

#After establishing our main hypothesis above, we will now start checking the normality assumption, setting
# its hypothesis up as follows:

# Ho: There is no statistically significant difference between the sample distribution and the normal distribution. (Normal or Gaussian distribution)
# Ha: There is a statistically significant difference between the sample distribution and the normal distribution. (Non-Normal distribution)

# We will test the assumption by the Shapiro-Wilk test for each group.

test_stat, pvalue = shapiro(tips.loc[tips["smoker"]=="Yes", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Test Stat = 0.9367, p-value = 0.0002

test_stat, pvalue = shapiro(tips.loc[tips["smoker"]=="No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Test Stat = 0.9045, p-value = 0.0000

# Since p-values are lower than 0.05, Ho hypothesis is rejected for both smokers and non-smokers. This simply means
# that the data is not normally distributed. In other words, there is a statistically significant difference between
# the sample distribution and the normal distribution.

# In this case, where the assumption of normality is not met, we should skip testing the assumption of variance homogeneity,
# and directly employ non-parametric Mann-Whitney U Test.

##############
# Non-parametric Mann-Whitney U Test
###

test_stat, pvalue = mannwhitneyu(tips.loc[tips["smoker"]=="Yes", "total_bill"],
                                 tips.loc[tips["smoker"]=="No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Test Stat = 6511.5000, p-value = 0.1707

# According to the result of the Mann-Whitney U-Test, where p-value is greater than 0.05, we cannot reject
# the null hypothesis (Ho) which says there is no statistically significant difference between the average bills paid by
# smokers and non-smokers. In other words, we cannot say statistically that those who smoke pay more than those who do not.

# Now the manager can make her decision based on statistical results.
