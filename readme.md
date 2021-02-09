Hi :wave:,

In this project we have HR data of a company. A study is requested from us to predict which employee will churn by using this data. The HR dataset has 14,999 samples. In the given dataset, we have two types of employee one who stayed and another who left the company.

![Churn](https://github.com/M-Rasit/Employee-Churn-Project/blob/master/images/churn.png?raw=true)


This is an imbalanced dataset and before creating machine learning algorithms, I performed data analysis on dataset and figured out important insights.

:pen: People who have low satisfaction level have most churn rate.

:pen: Employee who have low last evaluation point prefer to stay with company. Company lost employees from normal and very high class in their last evaluation.

:pen: People who work more average hours than other employees quit job.

:pen: Technical department has the most number of employee who work more than 300 hours in month. All of these employees churn from their company. 

:pen: HR department has the most churn rate.

:pen: Employees who works with low salary have higher churn rate.

:pen: Most of the employees who had a work accident, preferred to stay in company.

:pen: Promoted employees have lower churn rate.

![Satisfaction Level](https://github.com/M-Rasit/Employee-Churn-Project/blob/master/images/satisfaction%20by%20churn.png?raw=true)

![Average Monthly Hours](https://github.com/M-Rasit/Employee-Churn-Project/blob/master/images/Average%20Monthly%20Hours.png?raw=true)

![Time Spend in Company](https://github.com/M-Rasit/Employee-Churn-Project/blob/master/images/time_spend_company.png?raw=true)

I used Gradient Boosting, Random Forest and K Nearest Neighbors algorithms. I used RandomizedSearch and cross validation for getting best results truly.

![Models](https://github.com/M-Rasit/Employee-Churn-Project/blob/master/images/models.png?raw=true)

Finally I created a streamlit app. This app can predicts whether a single employee will churn or not. Moreover, it can analyze and visualize dataset. Also, it can calculate prediction probability and get more loyal or the opposite employees.

Thank you for your time:tulip:
