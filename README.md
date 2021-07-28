# Custumer-Churn
Customer Churn Analysis in R: Logistic, Classification Tree, XGBoost, Random Forest.

# Content:
1. Preprocessing & Data cleaning
2. Exploratory Data Analysis (EDA)
3. Feature selection & Chi-square Test
4. Predictive Models: Logistic, Classification Tree, XGBoost, Random Forest
5. Compare Models’ Performance
6. Feature Importance

# Code:
https://github.com/trajceskijovan/Custumer-Churn/blob/main/Customer%20Churn%20Analysis.R

# Context:
This analysis focuses on the behavior of bank customers who are more likely to leave the bank (i.e. close their bank account). 
The goal here is to identify the behavior of customers through Exploratory Data Analysis and later on use predictive analytics techniques to determine the customers who are most likely to churn (leave).

# EDA

    - CreditScore: from 350 to 850
    - Geography:France, Germany and Spain
    - Age: from 18 to 92
    - Tenure: how long customer has stayed with the bank
    - Balance: the amount of money available for withdrawal
    - NumOfProducts: number of products customers use in the bank
    - IsActiveMember: 0,1 -> Inactive, Active
    - EstimatedSalary: customer’s annual salary
    - Exited: whether the customer has churned (closed the bank account) where 0,1 -> Stay, Churn

# No NA`s:
![](samples/1.png)

# Target [ Stay (0), Churn/Leave(1) ]:
![](samples/2.png)

# Distribution - Continuous Variables:
![](samples/3.png)

1. Age is a bit right-skewed
2. Balance is fairly normal distributed
3. Most credit scores are above 600, it is possible that high quality customers will churn

# Correlation Matrix
![](samples/4.png)

1. No high correlation between the continuous variables (i.e. no multicollinearity)
2. We will keep all of the continuous variables

# Distribution - Categorical Variables:
![](samples/5.png)

1. More male customers than females
2. Customers are mostly from France
3. Most customers have the bank’s credit card
4. Almost equal number of active and non-active members, not a very good sign
5. Most customers use one or two kind of products, with a very few use three or four products
6. Almost equal number of customers in different tenure groups, except 0 and 10.

# Variables Deep Dive:

# AGE
![](samples/6.png)

1. Non-churned customers have a right-skewed distribution (tend to be young)
2. Outliers above 60 years old - perhaps our stable customers
3. Churned customers are mostly around 40 to 50. They might need to switch to other banking service for retirement purpose or whole family
4. We cab see very clear difference between these two groups

# BALANCE
![](samples/7.png)

1. Distribution of these two groups is similar
2. Surprisingly some non-churned customers have lower balance than churned customers

# CREDIT SCORE
![](samples/8.png)

1. Similar distribution
2. Some customers with extremely low credit score (on the left tail) as well as with high credit score also churned
3. Indicates that really low and high quality customer can easily churn than the average quality customer





