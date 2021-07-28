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

Target (Stay, Churn/Leave):
![](samples/1.png)
