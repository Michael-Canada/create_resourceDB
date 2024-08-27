import pandas as pd
from pandas import read_csv
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor


'''
This file contains the code for the economic model hypothesis testing project.
It runs on a made-up dataset that contains 1000 rows and 190 columns.
Most of the columns are numerical, and some are categorical, to simulate a real-world dataset.
This code is generic, and will be used to find the best model that can predict the target variable 'y', which could be a real-world economic indicator such as heat-rate, etc..
We explore two models: Linear and XGBoost, and compare between them to find the best model. "Best model" is defined as the model that has the lowest Mean Squared Error (MSE) on unseen data.
Additionally, we check if the model overfits the data. This is done by ensuring that the MSE of the in-sample data is within 2 standard deviations of the MSE of the out-of-sample data.

The steps in the code are:
1. Preprocess the data: 
    - impute missing values
    - scale numerical predictors
    - encode categorical predictors
    - remove bias from the target
2. Features selection (4 out of 190 predictors)
3. Build models on reduced data
4. Evaluate the models:
    - check if the model overfits the data
5. Report on the R-squared metric of the model
6. Make predictions



'''

random_state = 2    # to allow reproduction of results


def replace_inf_with_nan(X):
    return np.where(np.isinf(X), np.nan, X)

# Preprocess input data
def preprocess(path_train_data, path_test_data, target_title = 'y', raw_training_data_with_target = None, raw_testing_data = None, DEBUGGING = True):
    """
    Preprocessing the training and testing data and removing bias from the target:
    1. Impute missing values
    2. Scale numerical predictors
    3. Encode Categorical predictors
    4. Remove bias from y (the target)

    Inputs: 
        paths of the training and testing data
    Returs: 
        Training and testing data after preprocessing
        y (the target) after removing bias
        y's bias

    """
    
    # read and prepare datasets
    if not path_train_data is None:
        raw_training_data_with_target = read_csv(path_train_data)
        raw_testing_data = read_csv(path_test_data)

        raw_training_data = raw_training_data_with_target.drop(columns = 'y')
    else:
        raw_training_data = raw_training_data_with_target.drop(columns = target_title)

    y = raw_training_data_with_target[target_title].values

    # if 90% of the values in y are the same then return
    if raw_training_data_with_target[target_title].value_counts().max() > 0.9 * len(raw_training_data_with_target):
        return None, None, None, None, None, None, None

    # eliminate rows where the target value is nan
    raw_training_data_with_target = raw_training_data_with_target.dropna(subset=[target_title])

    # if number of rows is less than 20 then return
    if len(raw_training_data_with_target) < 20:
        return None, None, None, None, None, None, None
    
    # for debugging: training and testing datasets, prior to preprocessing
    # raw_train = raw_training_data.values
    # raw_test = raw_testing_data.values

    # remove bias from y:
    target_bias = np.mean(y)
    target_std = np.std(y)
    y = y - target_bias
    y = y / target_std

    if DEBUGGING:
        # remove columns that have nan in them:
        raw_training_data = raw_training_data.dropna(axis=1)

    # find numerical columns and their indices
    numerical_cols = raw_training_data.select_dtypes(include=['number']).columns
    numeric_indices = [raw_training_data.columns.get_loc(col) for col in numerical_cols]

    #eliminate columns that are not numerical
    raw_training_data_orig = raw_training_data.copy()
    raw_testing_data_orig = raw_testing_data.copy()
    # raw_training_data = raw_training_data[numerical_cols]
    # raw_testing_data = raw_testing_data[numerical_cols]

    # eliminate columns that only have one value
    for col in raw_training_data.columns:
        if col in numerical_cols:
            if len(raw_training_data[col].unique()) == 1:
                raw_training_data = raw_training_data.drop(columns = col)
                raw_testing_data = raw_testing_data.drop(columns = col)
    
    # eliminate columns that have more than 90% of the values are the same
    for col in raw_training_data.columns:
        if col in numerical_cols:
            if raw_training_data[col].value_counts().max() > 0.9 * len(raw_training_data):
                raw_training_data = raw_training_data.drop(columns = col)
                raw_testing_data = raw_testing_data.drop(columns = col)

    # eliminate columns that have more than 90% missing values
    for col in raw_training_data.columns:
        if col in numerical_cols:        
            if raw_training_data[col].isnull().sum() > 0.9 * len(raw_training_data):
                raw_training_data = raw_training_data.drop(columns = col)
                raw_testing_data = raw_testing_data.drop(columns = col)

    # find numerical columns and their indices
    numerical_cols = raw_training_data.select_dtypes(include=['number']).columns
    numeric_indices = [raw_training_data.columns.get_loc(col) for col in numerical_cols]

    new_training_cols = []
    new_testing_cols = []

    # add new features for numerical columns
    for col in numerical_cols:
        for col2 in numerical_cols:
            new_training_cols.append(pd.Series(raw_training_data.loc[:, col] * raw_training_data.loc[:, col2], name=col + '___' + col2))
            new_testing_cols.append(pd.Series(raw_testing_data.loc[:, col] * raw_testing_data.loc[:, col2], name=col + '___' + col2))
        new_training_cols.append(pd.Series(raw_training_data.loc[:, col] ** .5, name=col + '__sqrt'))
        new_testing_cols.append(pd.Series(raw_testing_data.loc[:, col] ** .5, name=col + '__sqrt'))
        new_training_cols.append(pd.Series(np.log(raw_training_data.loc[:, col]), name=col + '__log'))
        new_testing_cols.append(pd.Series(np.log(raw_testing_data.loc[:, col]), name=col + '__log'))

    raw_training_data = pd.concat([raw_training_data] + new_training_cols, axis=1)
    raw_testing_data = pd.concat([raw_testing_data] + new_testing_cols, axis=1)
        
    # After adding features we have to find numerical columns and their indices again
    numerical_cols = raw_training_data.select_dtypes(include=['number']).columns
    numeric_indices = [raw_training_data.columns.get_loc(col) for col in numerical_cols]

    # find categorical columns and their indices
    categorical_cols = raw_training_data.select_dtypes(include=['object']).columns
    categorical_indices = [raw_training_data.columns.get_loc(col) for col in categorical_cols]

    # move all the categorical columns to the end
    cols_to_move = ['prime_mover_code']  # replace with your column names
    # cols_to_move = ['category']  # replace with your column names
    cols = [col for col in raw_testing_data if col not in cols_to_move] + cols_to_move
    raw_training_data = raw_training_data[cols]
    raw_testing_data = raw_testing_data[cols]

    # get rid of all the categorical columns except for prime_mover_code
    for col in categorical_cols:
        if col != 'prime_mover_code0':
        # if col != 'category':    
            raw_training_data = raw_training_data.drop(columns = col)
            raw_testing_data = raw_testing_data.drop(columns = col)

    # find categorical columns and their indices
    categorical_cols = raw_training_data.select_dtypes(include=['object']).columns
    categorical_indices = [raw_training_data.columns.get_loc(col) for col in categorical_cols]

    # find bulean columns and their indices
    boolean_cols = raw_training_data.select_dtypes(include=['bool']).columns
    boolean_indices = [raw_training_data.columns.get_loc(col) for col in boolean_cols]

    # # convert bolean columns to numerical. True will be converted to 1 and False to 0
    if boolean_cols.size > 0:
        raw_training_data[boolean_cols] = raw_training_data[boolean_cols].astype(int)
        raw_testing_data[boolean_cols] = raw_testing_data[boolean_cols].astype(int)

    # remove columns that are 'bool' objects
    # raw_training_data = raw_training_data.drop(columns = boolean_cols)
    # raw_testing_data = raw_testing_data.drop(columns = boolean_cols)
    

    # Store the original column orders
    # original_columns = numeric_indices # + categorical_indices + boolean_indices
    original_columns = numeric_indices + categorical_indices + boolean_indices

    # Preprocessing pipelines for numerical columns
    numerical_pipeline = Pipeline([
        ('inf_replacer', FunctionTransformer(replace_inf_with_nan)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
        ])

    # Preprocessing pipelines for categorical columns
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # ('encoder', OneHotEncoder(handle_unknown='ignore')) 
        ('encoder', OrdinalEncoder()) 
        ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols) ])

    




    # Preprocess the input data (impute and scale)
    processed_training_data = preprocessor.fit_transform(raw_training_data)
    processed_testing_data = preprocessor.transform(raw_testing_data)
    
    # bring processed features to original order
    # processed_training_data = processed_training_data[:, np.argsort(original_columns)]
    # processed_testing_data = processed_testing_data[:, np.argsort(original_columns)]

    all_titles_of_features = raw_training_data.columns

    return processed_training_data, processed_testing_data, y, target_bias, target_std, numeric_indices, all_titles_of_features


# feature selection
def select_features(Chosen_model, X_train, y, X_test):
    """
    Select predictors that are of higher importance.

    Inputs:
        1. The chosen model
        2. Training data
        3. Target data
        4. Testing data
    Returns:
        1. subset of training data
        2. subset of testing data
        3. Indices of the predictors chosen

    Hyperparameters:
        The max allowed number of predictors to keep is hardcoded to 4 ('max_features')
    """

    # specify the model to be used and hyperparams
    model = SelectFromModel(Chosen_model, max_features=7)
    
    # learn relationship from training data
    model.fit(X_train, y)

    # transform train input data
    Chosen_features = model.transform(X_train)
    
    # transform test input data
    test_chosen_features = model.transform(X_test)

    # find the indices of the predictors that are kept
    chosen_indices = [ind for ind,v in enumerate(model.get_support()) if v == True]

    return Chosen_features, test_chosen_features, chosen_indices

# define the model: Linear
def build_linear_model(dataset, y):
    """
    Define a linear model and its evaluation method.
    The evaluation is based on k-fold cross validation, where k is hardcoded to 5.
    In order to gather enough data and look at the statistics of the evaluation procedure, this is repeated 6 times.
    The metric is MSE.

    Inputs:
        1. the training dataset
        2. the target
        3. fixed random state (so output can be reproduced)
    Returns:
        1. Scores of the evaluation (30 of them)
        2. The linear model
    """

    # define model and hyperparameters
    model = LinearRegression()  #choose hyperparameters using a grid search across a range of values or pipeline

    # 6 repeats of 5-fold cross-validation, creating 30 samples of MSE
    cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=random_state)

    # Evaluation:
    scores = cross_val_score(model, dataset, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    # scores to be positive
    scores = np.absolute(scores)

    return scores, model

# define the model: XGBoost
def build_XGBooster_model(dataset, y):
    """
    Define an XGBoost Regression model and its evaluation method.
    The evaluation is based on k-fold cross validation, where k is hardcoded to 5.
    In order to gather enough data and look at the statistics of the evaluation procedure, this is repeated 6 times.
    The metric is MSE.

    Inputs:
        1. the training dataset
        2. the target
        3. fixed random state (so output can be reproduced)
    Returns:
        1. Scores of the evaluation (30 of them)
        2. The linear model
    """

    # define model and hyperparameters
    # model = XGBRegressor()  #play with hyperparameters using a grid search across a range of values (or pipeline)
    model = XGBRegressor(n_estimators=50, max_depth=3, eta=0.075, subsample=0.5)

    # 6 repeats of 5-fold cross-validation, creating 30 samples of MSE
    cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=random_state)

    # Evaluation
    scores = cross_val_score(model, dataset, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    # scores to be positive
    scores = np.absolute(scores)

    return scores, model


def build_RandomForest_model(dataset, y):
    """
    Define a RandomForest Regression model and its evaluation method.
    The evaluation is based on k-fold cross validation, where k is hardcoded to 5.
    In order to gather enough data and look at the statistics of the evaluation procedure, this is repeated 6 times.
    The metric is MSE.

    Inputs:
        1. the training dataset
        2. the target
        3. fixed random state (so output can be reproduced)
    Returns:
        1. Scores of the evaluation (30 of them)
        2. The linear model
    """

    # define model and hyperparameters
    model = RandomForestRegressor(n_estimators=50, max_depth=3, n_jobs=-1)


    # 6 repeats of 5-fold cross-validation, creating 30 samples of MSE
    cv = RepeatedKFold(n_splits=5, n_repeats=6, random_state=random_state)

    # Evaluation
    scores = cross_val_score(model, dataset, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    # scores to be positive
    scores = np.absolute(scores)

    return scores, model






# check if the model overfits the data
def check_model_performance(selected_train_data_XGB, selected_train_data_linear, scores_reduced_XGB, scores_reduced_linear, model_XGB, model_linear, target):
    """
    Check if the model overfits the data.

    Inputs:
        param to print intermediate MSEs (for debugging purposes)
    Returns:
        strings that report on findings
    
    Details:
    Model performance is poor when one of two conditions are met:
        1. The model overfits the data
        2. The model is biased, i.e. the prediction errors are not zero-mean
    
        In this function we only check for model overfit
    
    Overfitting is tested by comparing the MSE of a model that is run on seen data ("in-sample") to the MSE of a model that is run on unseen data ("out of sample"). 
    The concept is that MSE of in-sample data that is much better than the MSE of out-of-sample data indicates that the model was overfitted on the training data.
    """

    print_output = True

    # make a prediction on entire training dataset
    yhat_reduced_XGB = model_XGB.predict(selected_train_data_XGB)
    MSE_XGB = np.mean((yhat_reduced_XGB - target)**2)
    yhat_reduced_linear = model_linear.predict(selected_train_data_linear)
    MSE_linear = np.mean((yhat_reduced_linear - target)**2)

    # print MSE of in-sample and out-of-sample predictions (for development purposes)
    if print_output:
        print('\nline 5: MSE score linear: seen data:      %.3e' % (MSE_linear))
        print('line 6: MSE score linear: unseen data:    %.3e (std: %.3e)' % (np.mean(scores_reduced_linear), np.std(scores_reduced_linear)))
        print('\nline 7: MSE score XGB: seen data:         %.3e' % (MSE_XGB))
        print('line 8: MSE score XGB: unseen data:       %.3e (std: %.3e) \n' % (np.mean(scores_reduced_XGB), np.std(scores_reduced_XGB)))

    # check for overfitting:
    # If the MSE of the in-sample fits within 2 standard deviations of the MSE of the out-of-sample then no overfitting
    if np.mean(scores_reduced_XGB) -2 * np.std(scores_reduced_XGB) <= MSE_XGB <= np.mean(scores_reduced_XGB) + 2 * np.std(scores_reduced_XGB):
        result_XGB = 'line 9:  GOOD: XGBoost NOT overfitting'
    else:
        result_XGB = 'line 9:  Bad:  XGB OVERFITTING !!'
    
    # check for overfitting:
    # If the MSE of the in-sample fits within 2 standard deviations of the MSE of the out-of-sample then no overfitting
    if np.mean(scores_reduced_linear) -2 * np.std(scores_reduced_linear) < MSE_linear < np.mean(scores_reduced_linear) + 2 * np.std(scores_reduced_linear):
        result_linear = 'line 10: GOOD: linear NOT overfitting'
    else:
        result_linear = 'line 10: Bad:  linear overfitting'

    return result_XGB, result_linear

def report_R_squared(target, scores_reduced_linear, scores_reduced_XGB):    
    """
    Report on the R-squared metric of the model
    """

    if np.std(target) > 0:
        print('line 11: R-squared_linear: %.3f '% (1 - np.mean(scores_reduced_linear) / np.std(target)**2))
        print('line 12: R-squared_XGB: %.3f ' %  (1 - np.mean(scores_reduced_XGB) / np.std(target)**2))
    else:
        print('The target values are constant and prediction does not require ML. Dealing with this is outside the scope of this project, and this potential issue will be ignored for now')

    return

def create_benchmark():
    """
    Benhmark the performance of linear and XGBoost models on the entire training dataset

    details:
    The function creates a benchmark of how well a linear and XGBoost models can capture the relationship between the predictors and the target.
    The intent is to use the benchmark as a baseline for improved models.
    The performance of the model is measured by the MSE which is the result of evaluation using k-fold cross-validation (which is implemented in the func 'build_XGBooster_model').
    For the purpose of creating a benchmark, the models run on the entire training dataset.

    Returns:
        1. colletion of MSE scores for XGBoost
        2. colletion of MSE scores for linear model    
    """
    # define models (linear and XGBoost), in which all the training dataset is used
    scores_linear, model_linear = build_linear_model(processed_training_data, target)
    scores_XGB, model_XGB = build_XGBooster_model(processed_training_data, target)
    
    return scores_XGB, scores_linear

def build_model_on_reduced_data(chosen_model, processed_training_data, target, processed_testing_data):

    if chosen_model == 'XGB':
        # Select iportant features (XGBoost), build and run model
        use_this_model = XGBRegressor()
        # use_this_model = XGBRegressor(n_estimators=50, max_depth=3, eta=0.075, subsample=0.5)
        # use_this_model = XGBRegressor(n_estimators=50, max_depth=3, tree_method='gpu_hist', early_stopping_rounds=2, eta=0.1, subsample=0.8)
        # use_this_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
        selected_train_data_XGB, selected_test_data_XGB, chosen_indices_XGB = select_features(use_this_model, processed_training_data, target, processed_testing_data)
        print('line 4: chosen indices XGB:              ', chosen_indices_XGB)
        scores_reduced_XGB, model_XGB = build_XGBooster_model(selected_train_data_XGB, target)
        model_XGB.fit(selected_train_data_XGB, target)
        return selected_train_data_XGB, model_XGB, scores_reduced_XGB, selected_test_data_XGB, chosen_indices_XGB
    elif chosen_model == 'RandomForest':
        # Select iportant features, build and run model
        use_this_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
        selected_train_data, selected_test_data, chosen_indices = select_features(use_this_model, processed_training_data, target, processed_testing_data)
        print('line 4: chosen indices RandomForest:              ', chosen_indices)
        scores_reduced_RF, model_RF = build_RandomForest_model(selected_train_data, target)
        model_RF.fit(selected_train_data, target)
        return selected_train_data, model_RF, scores_reduced_RF, selected_test_data, chosen_indices
    elif chosen_model == 'linear':
        # Select iportant features (linear), build and run model
        selected_train_data_linear, selected_test_data_linear, chosen_indices_linear = select_features(LinearRegression(), processed_training_data, target, processed_testing_data)
        print('line 3: chosen indices Linear:           ', chosen_indices_linear)
        scores_reduced_linear, model_linear = build_linear_model(selected_train_data_linear, target)
        model_linear.fit(selected_train_data_linear, target)
        return selected_train_data_linear, model_linear, scores_reduced_linear, selected_test_data_linear, chosen_indices_linear
    else:
        print('Not supposed to be here. Dealing with this error is outside of the scope of this project')
    
    return None

def make_predictions(chosen_model, target_bias, target_std, selected_test_data_linear, selected_test_data_XGB, model_XGB, model_linear):
    # make a prediction
    if chosen_model == 'XGB':
        yhat = model_XGB.predict(selected_test_data_XGB)
        # add the target bias
        yhat *= target_std
        yhat += target_bias
        return yhat

    elif chosen_model == 'linear':
        yhat = model_linear.predict(selected_test_data_linear)
        # add the target bias
        yhat *= target_std
        yhat += target_bias
        return yhat

    else:
        print('Dealing with this Exception is outside the scope of this project, and this potential issue will be ignored for now')
    
    return None


# def calculate_linear_feasibility():
#     """
#     Feasibility check for a linear model. 
    
#     Details:
#     The concept is to solve a linear equation that maps the predictors to the target ('y'). 
#     In the linear equation Ax=y, A is the predictors and y is the target. The Mean Squared Errors of the calculation indicate the R-squared measure of the solution. R-squared that is lower than 0.5 indicate that a linear model is unfeasible!
#     """

#     numeric_predictors = processed_training_data[:, numeric_indices]  #indices of numeric predictors
#     y_variance = np.var(target)                                       # variance of the target 'y'
#     num_of_reads = np.shape(processed_training_data)[0]               # 1000 read in the training dataset
    
#     # Sum of quared errors of the linear equation Ax=y (A: predictors, y: target, x: the linear solution):
#     squared_errors = np.linalg.lstsq(numeric_predictors, target, rcond = 0.01)[1]

#     # the mean squared error:
#     mean_squared_errors = squared_errors / num_of_reads               #average of the squared error
    
#     # calculate R-squared
#     best_R_squared_linear_approach = 1 - mean_squared_errors / y_variance

#     return best_R_squared_linear_approach

if __name__ == '__main__':
    # Load input data
    path_train_data = '/Users/michael.simantov/Documents/trainingData.csv'
    path_test_data  = '/Users/michael.simantov/Documents/testData.csv'

    print('\nFindings Regarding Model Selection')
    print('--------------------------------------')

    # Preprocess raw data, to impute NaN, scale numerical features and remove target bias
    DEBUGGING = False
    processed_training_data, processed_testing_data, target, target_bias, target_std, numeric_indices, all_titles_of_features = preprocess(path_train_data, path_test_data, DEBUGGING = DEBUGGING)

    # create models on reduced train datasets
    selected_train_data_linear, model_linear, scores_reduced_linear, selected_test_data_linear, chosen_indices_XGB = build_model_on_reduced_data('linear', processed_training_data, target, processed_testing_data)
    selected_train_data_XGB, model_XGB, scores_reduced_XGB, selected_test_data_XGB, chosen_indices_linear = build_model_on_reduced_data('XGB', processed_training_data, target, processed_testing_data)

    # Check overfitting of models that were created on reduced data
    result_XGB, result_linear = check_model_performance(selected_train_data_XGB, selected_train_data_linear, scores_reduced_XGB, scores_reduced_linear, model_XGB, model_linear, target)
    print(result_XGB)
    print(result_linear)

    # Report R-squared score of the models
    report_R_squared(target, scores_reduced_linear, scores_reduced_XGB)

    # Predict:
    print('\nPrediction Results of the Linear Model')
    print(' --------------------------------------')
    print('   No.   Predicted_Value')

    Use_linear_model = True
    if Use_linear_model:
        prediction_linear = make_predictions('linear', target_bias, target_std, selected_test_data_linear, selected_test_data_XGB, model_XGB, model_linear)
    else:
        prediction_XGB = make_predictions('XGB', target_bias, target_std, selected_test_data_linear, selected_test_data_XGB, model_XGB, model_linear)

    number_of_predicted_values_to_print = 10

    for ind,val in enumerate(prediction_linear[:number_of_predicted_values_to_print]):
        print(f"{str(ind+1):^9} " + str(val))

