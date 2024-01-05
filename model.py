import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data=pd.read_csv(r"C:\Users\jenan\Downloads\DataCoSupplyChainDataset.csv\DataCoSupplyChainDataset.csv",encoding_errors="ignore")
print("loaded data")

data=data.drop(['Customer Lname','Customer Zipcode','Order Zipcode','Product Description'],axis=1)

#Removing duplicate and unwanted columns
data=data.drop(['Sales per customer','Category Id','Customer Email','Customer Fname','Customer Id', 'Customer Password', 
              'Customer Street','Department Id','Latitude','Longitude','Order Customer Id',
              'Order Id', 'Order Item Cardprod Id','Order Item Id','Order Profit Per Order',
              'Product Card Id', 'Product Category Id','Product Image','Order Item Product Price',
              'Product Status','Delivery Status'],axis=1)

#Feature engineering
from datetime import datetime as dt
data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])
data['Order date']=data["order date (DateOrders)"].dt.day
data["order year"] = data["order date (DateOrders)"].dt.year
data["order month"] = data["order date (DateOrders)"].dt.month
data["order day of the week"] = data["order date (DateOrders)"].dt.weekday
data["is weekend"] = data["order day of the week"] > 4
data["order hour"] = data["order date (DateOrders)"].dt.hour

data['Is_Late'] = np.where(data['Days for shipping (real)'] > data['Days for shipment (scheduled)'],'YES','NO')

#Label encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
to_encode=['Type','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Department Name',
          'Market','Order City','Order Country','Order Region','Order State','Order Status','Product Name','Shipping Mode','is weekend']
for col in to_encode:
    data[col]=le.fit_transform(data[col])

data=data.drop(['order date (DateOrders)','shipping date (DateOrders)','Late_delivery_risk','Days for shipping (real)', 
                'Days for shipment (scheduled)'],axis=1)

data=data.drop(['Order Status','Order Item Profit Ratio','Order Item Discount Rate','Sales'],axis=1)

x=data[['Type','Category Name','Customer Country','Customer Segment','Customer State','Department Name','Market',
       'Order Country','Order Item Quantity','Order Region','Product Price','Shipping Mode',
       'order year', 'order month','Order date',
       'order day of the week', 'is weekend', 'order hour']]
y=data['Is_Late']

#Standardizing the features
from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
X=std_scaler.fit_transform(x)
X=pd.DataFrame(x)

#Splitting the data in such a way that 75% data is for training and 25% is for testing
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)

from sklearn.ensemble import ExtraTreesClassifier
xt_cls=ExtraTreesClassifier()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

# Initialize GridSearchCV with random forest model and parameter grid
grid_search_xt = GridSearchCV(xt_cls, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search_xt.fit(x_train, y_train)


# Get the best parameters from the grid search
best_params = grid_search_xt.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model to make predictions on the test set
best_model = grid_search_xt.best_estimator_
y_pred_xt_grid = best_model.predict(x_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred_xt_grid)
print("Accuracy on Test Set:", accuracy)

#Creating pickle file for SVM model
pickle_file="xt_model.pickle"
with open(pickle_file,'wb') as file:
    pickle.dump(grid_search_xt,file)

print(f"Best model has been pickled and saved to {pickle_file}")