import pickle
from IPython.display import display, HTML
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils

# Setup data.
warnings.filterwarnings("ignore")
display(HTML("<style>pre { white-space: pre !important; }</style>"))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('Aemf1.csv', header=0)

# Convert string columns
unique_val_temp = df['City'].unique()
temp_dict = dict(zip(unique_val_temp, range(len(unique_val_temp))))
print(temp_dict)
df = df.replace(temp_dict)

unique_val_temp = df['Day'].unique()
temp_dict = dict(zip(unique_val_temp, range(len(unique_val_temp))))
df = df.replace(temp_dict)

unique_val_temp = df['Room Type'].unique()
temp_dict = dict(zip(unique_val_temp, range(len(unique_val_temp))))
print(temp_dict)
df = df.replace(temp_dict)

unique_val_temp = df['Shared Room'].unique()
temp_dict = dict(zip(unique_val_temp, range(len(unique_val_temp))))
print(temp_dict)
df = df.replace(temp_dict)
# print(list(df.columns))

# X = df.copy()
# X = X.drop(columns=['Guest Satisfaction', 'Attraction Index', 'Normalised Restaurant Index', 'Person Capacity',
#                     'Day', 'Price', 'Metro Distance (km)', 'City Center (km)'])
X = df[['Cleanliness Rating', 'Normalised Restaurant Index', 'Normalised Attraction Index', 'City', 'Bedrooms', 'Business', 'Superhost', 'Room Type', 'Private Room', 'Multiple Rooms']]
# # print(list(X.columns))
y = df[['Guest Satisfaction']]
# X = sm.add_constant(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    test_size=0.25, random_state=100)

model = sm.OLS(y_train, x_train).fit()
# print(model.summary())
# predictions = model.predict(x_test)
# print('Root Mean Squared Error:',
#       np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Save the model.
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)

#
# print(x_test)
# y_pred = loadedModel.predict(x_test)


# # feature_list = list(X.columns)
# feature_list = ['Cleanliness Rating', 'Normalised Restaurant Index', 'Normalised Attraction Index', 'City', 'Bedrooms', 'Business', 'Superhost', 'Room Type', 'Private Room', 'Multiple Rooms']
# labels = np.array(y)
# features = np.array(X[feature_list])
#
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
# rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# model = rf.fit(train_features, train_labels)
# predictions = rf.predict(test_features)
# errors = abs(predictions - test_labels)
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('------Feature Importance Selection------')
# print('Accuracy:', round(accuracy, 2), '%.')
# mse = mean_squared_error(test_labels, predictions)
# print('RMSE:', np.sqrt(mse))
#
# # Save the model.
# with open('model_pkl', 'wb') as files:
#     pickle.dump(model, files)
#
# # load saved model
# with open('model_pkl', 'rb') as f:
#     loadedModel = pickle.load(f)




# importances = list(rf.feature_importances_)


# def showFeatureImportances(importances, feature_list):
#     dfImportance = pd.DataFrame()
#     for i in range(0, len(importances)):
#         dfImportance = dfImportance.append({"importance":importances[i],
#                                             "feature":feature_list[i] },
#                                             ignore_index = True)
#     dfImportance = dfImportance.sort_values(by=['importance'],
#                                             ascending=False)
#     print(list(dfImportance['feature'][0:10]))
# showFeatureImportances(importances, feature_list)


# def getUnfitModels():
#     models = list()
#     models.append(ElasticNet())
#     models.append(SVR(gamma='scale'))
#     models.append(DecisionTreeRegressor())
#     models.append(RandomForestRegressor(n_estimators=200))
#     models.append(ExtraTreesRegressor(n_estimators=200))
#     return models
#
# def evaluateModel(y_test, predictions, model):
#     mse = mean_squared_error(y_test, predictions)
#     rmse = round(np.sqrt(mse),3)
#     print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)
#
#
# def fitBaseModels(X_train, y_train, X_test, models):
#     dfPredictions = pd.DataFrame()
#
#     # Fit base model and store its predictions in dataframe.
#     for i in range(0, len(models)):
#         models[i].fit(X_train, y_train)
#         predictions = models[i].predict(X_test)
#         colName = str(i)
#         # Add base model predictions to column of data frame.
#         dfPredictions[colName] = predictions
#     return dfPredictions, models
#
#
# def fitStackedModel(X, y):
#     model = LinearRegression()
#     model.fit(X, y)
#     return model
#
# # Split data into train, test and validation sets.
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)
#
# # Get base models.
# unfitModels = getUnfitModels()
#
# # Fit base and stacked models.
# dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
# stackedModel          = fitStackedModel(dfPredictions, y_val)
#
# # Evaluate base models with validation data.
# print('****** Stacked model - Linear Regression')
# print("\n** Evaluate Base Models **")
# dfValidationPredictions = pd.DataFrame()
# for i in range(0, len(models)):
#     predictions = models[i].predict(X_test)
#     colName = str(i)
#     dfValidationPredictions[colName] = predictions
#     evaluateModel(y_test, predictions, models[i])
#
# # Evaluate stacked model with validation data.
# stackedPredictions = stackedModel.predict(dfValidationPredictions)
# print("\n** Evaluate Stacked Model **")
# evaluateModel(y_test, stackedPredictions, stackedModel)
#
#
# from   sklearn.model_selection import train_test_split
# from   sklearn.linear_model    import LogisticRegression
#
# # Re-assign X with significant columns only after chi-square test.
# # Split data.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
#                                                  random_state=0)
#
#
# # Build logistic regression model and make predictions.
# logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear',
#                                    random_state=0)
# logisticModel.fit(X_train, y_train)
#
# predictions = logisticModel.predict(X_test)
# print('Root Mean Squared Error:',
#       np.sqrt(metrics.mean_squared_error(y_test, predictions)))
