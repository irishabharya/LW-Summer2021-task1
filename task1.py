import pandas as pd
import numpy

db = pd.read_csv("SalaryData.csv")

y=db['Salary']
x=db['YearsExperience']
x=x.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x,y)

years=int(input("Enter years of Experience to predict the salary: "))

result=model.predict([[years]])

print("\nPredicted Salary is : ", result)
