# Libraries

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
st.set_option('deprecation.showPyplotGlobalUse', False)

# Models and Variables

model = pickle.load(open("rfc_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))

# Sidebar



def main():
    
    # Sidebar
    
    st.sidebar.title("Find Single Employee Churn")
    
    satisfaction = st.sidebar.slider("Satisfaction Level", 0, 100, 1) / 100
    last_evaluation = st.sidebar.slider("Last Evaluation", 0, 100, 1) / 100
    number_project = st.sidebar.slider("Number of Projects", 0, 20, 1)
    monthly_hours = st.sidebar.slider("Average Monthly Hours", 0,500,1)
    time_spend = st.sidebar.slider("Time Spend in Company", 0,20,1)
    work_accident = st.sidebar.selectbox("Work Accident", ["No", "Yes"])
    promotion = st.sidebar.selectbox("Promoted in Last 5 Years", ["No", "Yes"])
    departments = st.sidebar.selectbox("Department", ['IT', 'RandD', 'Accounting', 'Hr', 'Management', 'Marketing', 'Product_mng', 'Sales',                                        'Support', 'Technical'])
    salary = st.sidebar.selectbox("Salary Status", ["Low", "Medium", "High"])
    df = sidebar_df(satisfaction, last_evaluation, number_project, monthly_hours, time_spend, work_accident, promotion, departments, salary)
    
    # Sidebar Prediction
    if st.sidebar.button("Analyze"):
        prediction = model.predict(df)[0]
        if prediction == 0:
            st.sidebar.success("Stay")
        else:
            st.sidebar.error("Churn")

    # Mainbar
    
    html_temp = """
                <div style="background-color:red;padding:1.5px">
                    <h1 style="color:white;text-align:center;">Churn Prediction ML App</h1>
                </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    
    
    df2 = load_data()
    if st.checkbox("Dataset Analyze"):
        select1 = st.selectbox("Please select a section:", ["", "Head", "Describe", "Visualization"])
     
        if select1 == "Head":
            st.table(df2.head())
        elif select1 == "Describe":
            select2 = st.selectbox("Please select value type:", ["", "Numerical", "Categorical"])
            if select2 == "Numerical":
                st.table(df2.describe())
            elif select2 == "Categorical":
                st.table(df2[["Departments ", "salary"]].describe())
        elif select1 == "Visualization":
            plt.title("Employee Count for Number of Projects by Status", c="blue", size=14)
            sns.countplot(x="number_project", hue="left", data=df2, palette="tab10")
            st.pyplot()
        
            plt.title("Employee Count for Year of Experience by Status", c="blue", size=14)
            sns.countplot(x="time_spend_company", hue="left", data=df2, palette="tab10")
            st.pyplot()
        
            plt.title("Employee Counts in Departments by Churn", c="blue", size=14)
            sns.countplot(x="Departments ", hue="left", data=df2, palette="tab10")
            st.pyplot()
        
            plt.title("Employee Counts of Salary Scale by Churn", c="blue", size=14)
            sns.countplot(x="salary", hue="left", data=df2, palette="tab10")
            st.pyplot()
        
            plt.title("Employee Count of Total Accident Number by Churn", c="blue", size=14)
            sns.countplot(x="Work_accident", hue="left", data=df2, palette="tab10")
            st.pyplot()

    elif st.checkbox("Employees From Dataset"):
        
        d = process()
        select3 = st.selectbox("Please select a section:", ["", "Find Random Employees", "Find Top N Loyal Employees", "Find Top N Employees with High Churn Probability"])
        
        if select3 == "Find Random Employees":
            random = st.number_input("Please enter number:", min_value=1, step=1)
            st.table(d.sample(random))
    
        elif select3 == "Find Top N Loyal Employees":
            loyal = st.number_input("Please enter number:", min_value=1, step=1)
            st.table(d.sort_values(by="prediction_proba", ascending=False).head(loyal))
            
        elif select3 == "Find Top N Employees with High Churn Probability":
            churn = st.number_input("Please enter number:", min_value=1, step=1)
            st.table(d.sort_values(by="prediction_proba").head(churn))
    
def sidebar_df(satisfaction, last_evaluation, number_project, monthly_hours, time_spend, work_accident, promotion, departments, salary):
    
    # Feature Engineering
    
    if work_accident == "Yes": work_accident = 1
    else: work_accident = 0

   
    if promotion == "Yes": promotion = 1
    else: promotion = 0

    
    # Creating DataFrame

    df = pd.DataFrame(data=[[satisfaction, last_evaluation, number_project, monthly_hours, time_spend, work_accident, promotion]], 
                      columns=['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',                                    'Work_accident','promotion_last_5years'])

    df_department = pd.DataFrame(data=[[0] * 10], columns=['IT', 'RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng',                                                                'sales', 'support', 'technical'])

    for i in df_department.columns:
        if i.title() == departments:
            df_department.loc[0,i] = 1
        
    # Scaling and Label Encoding

    df2 = df.join(df_department)
    if salary == "High": df2["salary"] = 3
    elif salary == "Medium": df2["salary"] = 2
    else: df2["salary"] = 1
    
    # Clustering
    
    cluster = kmeans.predict(df2)[0]
    df2["cluster_label"] = cluster
    
    arr = scaler.transform(df2)
    df3 = pd.DataFrame(arr, columns=df2.columns)
   
    return df3



def load_data():
    df = pd.read_csv("HR_Dataset.csv")
    return df        


def process():
    
    df = pd.read_csv("HR_Dataset.csv").drop("left", axis=1)
    df2 = df[df.columns[:-1]]

    # Get dummies
    df2 = df2.join(pd.get_dummies(df["Departments "]))
    df2.drop("Departments ", axis=1, inplace=True)

    # Label encoding
    df2["salary"] = df["salary"].map({"high":3, "medium":2, "low":1})
    
    # Clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df2)
    df2["cluster_label"] = kmeans.labels_

    scale = StandardScaler()
    sc = scale.fit_transform(df2)
    df3 = pd.DataFrame(sc, columns=df2.columns)
    pred_proba = [i[1] for i in model.predict_proba(df3)]
    df["prediction_proba"] = pred_proba

    return df

if __name__ == "__main__":
    main()

