

## **Table of Contents**
1. [Environment](#environment)
2. [Introduction](#introduction)
3. [Dataset Information](#dataset-information)
4. [Python Code Structure](#python-code-structure)
    - [1. Importing Required Libraries](#1-importing-required-libraries)
    - [2. Loading the Data](#2-loading-the-data)
    - [3. Data Preparation](#3-data-preparation)
    - [4. Independent and Dependent Variable Selection](#4-independent-and-dependent-variable-selection)
    - [5. Splitting the Data into Training and Testing Sets](#5-splitting-the-data-into-training-and-testing-sets)
    - [6. Developing a Decision Tree Model](#6-developing-a-decision-tree-model)
    - [7. Model Accuracy Assessment](#7-model-accuracy-assessment)
    - [8. Visualizing the Decision Tree Model](#8-visualizing-the-decision-tree-model)
    - [9. Feature Importance in Decision Tree](#9-feature-importance-in-decision-tree)
    - [10. Developing a Random Forest Model](#10-developing-a-random-forest-model)
    - [11. Feature Importance in Random Forest](#11-feature-importance-in-random-forest)
    - [12. Forecasting Net Profit using Both Models](#12-forecasting-net-profit-using-both-models)
    - [13. Creating Excel Files for Model Structures](#13-creating-excel-files-for-model-structures)
    - [14. Saving Model Visualizations in Excel](#14-saving-model-visualizations-in-excel)
5. [Conclusion](#conclusion)
6. [References](#references)

## **Environment**
Jupyter Notebook - Python 3 (ipykernel)

## **Introduction**
This analysis focuses on developing and evaluating predictive models using a franchises dataset containing sales records across different locations and business types. Our objective is to construct decision tree and random forest models capable of forecasting net profit by employing robust machine learning methodologies to uncover underlying patterns and relationships between counter sales, drive-through sales, customer traffic, and business characteristics on net profit.

## **Dataset Information**
The dataset, named `Franchises Dataset.xlsx`, contains data collected from 100 different franchises. The attributes in the dataset are:

- **Net Profit (million $)**
- **Counter Sales (million $)**
- **Drive-through Sales (million $)**
- **Number of customers visiting daily**
- **Business Type**
- **Location**

## **Python Code Structure**

### **1. Importing Required Libraries**
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
```

### **2. Loading the Data**
```python
# Load the dataset into a DataFrame
df = pd.read_excel(r'C:\Users\USER\Downloads\Franchises Dataset.xlsx')

# Display the first few rows of the DataFrame
display(df.head())
```

### **3. Data Preparation**
```python
# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Encode categorical variables
encoder = LabelEncoder()
df['Business Type'] = encoder.fit_transform(df['Business Type'])
df['Location'] = encoder.fit_transform(df['Location']) 

# Display the updated DataFrame
display(df.head())
```

### **4. Independent and Dependent Variable Selection**
```python
# Separate features and target variable
X = df.drop("Net Profit", axis=1)
y = df["Net Profit"]
```

### **5. Splitting the Data into Training and Testing Sets**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
```

### **6. Developing a Decision Tree Model**
```python
# Train the decision tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Assess the accuracy of the model
tree_predictions = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(y_test, tree_predictions)
```

### **7. Model Accuracy Assessment**
```python
# Create a data frame to present the results
results = pd.DataFrame({
    'Metric': ['MSE','RMSE','R^2 Score'],
    'Value': [tree_mse,tree_rmse,tree_r2]
})

# Print the DataFrame
display(results)
```

### **8. Visualizing the Decision Tree Model**
```python
from sklearn.tree import plot_tree

plt.figure(figsize=(50,40))
plot_tree(tree_model, filled=True, feature_names=X.columns)
plt.show()
```

### **9. Feature Importance in Decision Tree**
```python
# Simulate decision tree
feature_importances = tree_model.feature_importances_
sorted_indices = feature_importances.argsort()[::-1]
sorted_features = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Create a bar graph to visualize feature importance
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_features, sorted_importances, color='darkblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Decision Tree Feature Importances (Descending Order)')
plt.xticks(rotation=45)

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### **10. Developing a Random Forest Model**
```python
# Train the random forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Assess the accuracy of the model
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Model': ['Random Forest'],
    'MSE': [rf_mse],
    'RMSE': [rf_rmse],
    'R^2 Score': [rf_r2]
})

display(metrics_df)
```

### **11. Feature Importance in Random Forest**
```python
# Feature importance for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualize feature importance
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
bars = plt.bar(X.columns[indices], importances[indices], color='darkblue')
plt.xticks(rotation=45)

# Adding value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()
```

### **12. Forecasting Net Profit using Both Models**
```python
# Create a new data point with the given feature values
new_data = pd.DataFrame({
    'Counter Sales': [500000],
    'Drive-through Sales': [700000],
    'number of customers': [100],
    'Business Type': [2],  # Pizza store
    'Location': [0]  # Richmond
})

# Forecast net profit using decision tree model
dt_forecast = tree_model.predict(new_data)
print(f"Decision Tree Forecast: ${dt_forecast[0]:.2f} million")

# Forecast net profit using random forest model
rf_forecast = rf_model.predict(new_data)
print(f"Random Forest Forecast: ${rf_forecast[0]:.2f} million")
```

### **13. Creating Excel Files for Model Structures**
```python
import pandas as pd
from sklearn.tree import export_text
import openpyxl

# Export Decision Tree as text
tree_text = export_text(tree_model, feature_names=X_train.columns)
print("Decision Tree Text:")
print(tree_text)

# Export Random Forest as text
rf_text = ''
for index, estimator in enumerate(rf_model.estimators_):
    rf_text += f'Tree {index}:\n'
    rf_text += export_text(estimator, feature_names=X_train.columns) + '\n\n'
print("Random Forest Text:")
print(rf_text)

# Create a new Excel file
file_path = r'C:\Users\USER\Downloads\model_structures.xlsx'
workbook = openpyxl.Workbook()

# Add a worksheet for the decision tree
tree_sheet = workbook.create_sheet('Decision Tree')
tree_sheet['A1'] = tree_text

# Add a worksheet for the random forest
rf_sheet = workbook.create_sheet('Random Forest')
rf_sheet['A1'] = rf_text

# Save the workbook
workbook.save(file_path)
```

### **14. Saving Model Visualizations in Excel**
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import openpyxl
from openpyxl.drawing.image import Image

# Plot the decision tree
fig, ax = plt.subplots(figsize=(40, 20))
plot_tree(tree_model, filled=True, feature_names=X.columns, ax=ax)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='

tight')

# Load the Excel file
file_path = r'C:\Users\USER\Downloads\model_structures.xlsx'
workbook = openpyxl.load_workbook(file_path)

# Add the decision tree plot to the decision tree sheet
tree_sheet = workbook['Decision Tree']
img_tree = Image('decision_tree.png')
tree_sheet.add_image(img_tree, 'A2')

# Save the updated workbook
workbook.save(file_path)
```

## **Conclusion**
This project demonstrates the application of machine learning models to predict net profit for various franchise businesses based on sales data, customer traffic, business type, and location. The analysis includes both decision tree and random forest models, with their performance assessed through metrics like MSE, RMSE, and R^2 scores. The feature importance for both models is visualized to understand the impact of different variables. Additionally, a forecast is made for a new data point using both models, and the models' structures are saved and visualized in an Excel file.

## **References**
1. Fang, B., Yang, H., & Fan, L.-C. (2023). Vault predicting after implantable collamer lens implantation using random forest network based on different features in ultrasound biomicroscopy images. International Journal of Ophthalmology, 16(10), 1561–1567. https://doi.org/10.18240/ijo.2023.10.01 
2. Vens, C. (2013). Random Forest. Encyclopedia of Systems Biology, 1812–1813. https://doi.org/10.1007/978-1-4419-9863-7_612
3. Yang, F.-J. (2019). An extended idea about decision trees. 2019 International Conference on Computational Science and Computational Intelligence (CSCI). https://doi.org/10.1109/csci49370.2019.00068
4. Zhao, X., Wu, Y., & Yao, Y. (2023). A machine learning-based neural network for prediction of battery degradation and application in health management of rechargeable batteries. Electrochimica Acta, 466, 142698. https://doi.org/10.1016/j.electacta.2023.142698
