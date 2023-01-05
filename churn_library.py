'''

library doc string
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report

pd.set_option('display.max_columns', 99)



def import_data(pth):
    '''Rreturns dataframe for the csv found at pth

    input:
        pth: a path to the csv

    output:
        df: pandas dataframe
    '''
    print("import_data called")
    # df = pd.read_csv(r"./data/bank_data.csv", index_col=0,  nrows = 10)
    df = pd.read_csv(r"./data/bank_data.csv", index_col=0)
    return(df)



def perform_eda(df):
    '''Perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''
    print("perform_eda called")

    # Get high level stats
    print("\nDataframe has {0} rows and {1} columns".format(df.shape[0], df.shape[1]))
    print("\nView 10 random records\n", resample(df, n_samples=10, random_state=1234, replace=False))
    print("\nProportion of missing values in each column\n", df.isnull().sum() / df.shape[0])


    # Get a list of the numeric and caregorical feature names
    num_feature_names = df.select_dtypes(include='number').columns.tolist()
    num_feature_names.remove('CLIENTNUM')   # Remove the URN from the list
    cat_field_names = df.select_dtypes(include=['object','category']).columns.tolist()


    #Get distributions of numeric fields
    for feature in num_feature_names:
        sns.displot(df[feature], kde = True)
        plt.title("Distribution of {0}".format(feature))
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(r"./images/eda/{0}_distribution.png".format(feature))

    # Get distributions of categorical fields
    for feature in cat_field_names:
        sns.displot(df[feature])
        plt.title("Distribution of {0}".format(feature))
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(r"./images/eda/{0}_distribution.png".format(feature))

    # Get correlations of the numeric fields
    plt.figure(figsize=(20,10))
    sns.heatmap(df[num_feature_names].corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Numeric feature correlations")
    plt.savefig(r"./images/eda/Numeric_feature_correlations.png")



def derive_label(df, response = 'Churn'):
    '''Derive the target field. NB This assumes a binary classifier

    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new target field
    '''
    df[response] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def encoder_helper(df, cat_recode_lst, response = 'Churn'):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 16 from the notebook

    input:
            df: pandas dataframe
            cat_recode_lst: list of columns that contain categorical features to recode
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    print("encoder_helper called")

    for feature in cat_recode_lst:
        recode_map = df.groupby(feature).mean()[response]
        df["{0}_Churn".format(feature)] = df[feature].replace(recode_map)

    return df



def perform_feature_engineering(df, cat_recode_lst, features_lst, response = 'Churn'):
    '''
    input:
        df: pandas dataframe
        cat_recode_lst: list of columns that contain categorical features to recode
        features_lst: list of columns to include as model features
        response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data features
        X_test: X testing data features
        y_train: y training data labels
        y_test: y testing data labels
    '''
    print("perform_feature_engineering called")

    # Recode the categorical fields
    df = encoder_helper(df, cat_recode_lst)

    # Split into train and test datasets
    y = df[response]
    X = df[features_lst]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test



def classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf,
                                y_train_probs_lr, y_train_probs_rf,
                                y_test_probs_lr, y_test_probs_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions (classification) from logistic regression
            y_train_preds_rf: training predictions (classification) from random forest
            y_test_preds_lr: test predictions (classification) from logistic regression
            y_test_preds_rf: test predictions (classification) from random forest

            y_train_probs_lr: training predictions (probability) from logistic regression
            y_train_prob_rf: training predictions (probability) from random forest
            y_test_prob_lr: test predictions (probability) from logistic regression
            y_test_prob_rf: test predictions (probability) from random forest

    output:
             None
    '''
    print("classification_report_image called")


    # Classification report as text
    fp = open(r"./images/results/classification_report.txt", "w")
    fp.write('Random forest results')
    fp.write('\nTest results\n')
    fp.write(classification_report(y_test, y_test_preds_rf))
    fp.write('\nTrain results\n')
    fp.write(classification_report(y_train, y_train_preds_rf))
    fp.write('\n\nLogistic regression results')
    fp.write('\nTest results\n')
    fp.write(classification_report(y_test, y_test_preds_lr))
    fp.write('\nTrain results\n')
    fp.write(classification_report(y_train, y_train_preds_lr))
    fp.close()


    # Train ROC curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_predictions(y_train, y_train_probs_lr, name = 'LogReg', ax=ax)
    RocCurveDisplay.from_predictions(y_train, y_train_probs_rf, name = 'RandForest', ax=ax)
    plt.title("Train ROC")
    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    plt.savefig(r"./images/results/Train ROC curve.png")


    # Test ROC curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_predictions(y_test, y_test_probs_lr, name = 'LogReg', ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_test_probs_rf, name = 'RandForest', ax=ax)
    plt.title("Test ROC")
    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    plt.savefig(r"./images/results/Test ROC curve.png")



def feature_importance_plot(model, features_lst, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
        model: model object containing feature_importances_
        features_lst: list of columns to include as model features
        output_pth: path to store the figure

    output:
        None
    '''
    print("feature_importance_plot called")

    # Get feature importance
    importances = model.feature_importances_

    # Sort the features list to match the order
    indices = np.argsort(importances)[::-1]
    sorted_features_lst = [features_lst[i] for i in indices]

    # Plot the feature importance
    plt.figure(figsize=(15,15))
    plt.bar(range(len(sorted_features_lst)), importances[indices], tick_label = sorted_features_lst)
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.savefig(output_pth)



def train_models(X_train, X_test, y_train, y_test, features_lst):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              features_lst: list of columns to include as model features

    output:
              None
    '''
    print("train_models called")

    # Initialise the models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)


    # Define  the random forest hyperparameter search space
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],       # Auto has been depreciated
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train the models
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)


    # Get train and test scores (probability and classification)
    # Probability is needed for the RocCurveDisplay.from_predictions() call
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_train_probs_rf = cv_rfc.best_estimator_.predict_proba(X_train)[:,1]
    y_train_preds_lr = lrc.predict(X_train)
    y_train_probs_lr = lrc.predict_proba(X_train)[:, 1]
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_test_probs_rf = cv_rfc.best_estimator_.predict_proba(X_test)[:,1]
    y_test_preds_lr = lrc.predict(X_test)
    y_test_probs_lr = lrc.predict_proba(X_test)[:,1]


    # Get classification_reports and ROC curves
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf,
                                y_train_probs_lr, y_train_probs_rf,
                                y_test_probs_lr, y_test_probs_rf)


    # Get feature importances for random forest
    feature_importance_plot(cv_rfc.best_estimator_, features_lst, r"./images/results/RF feature importance.png")


    # Save models
    with open(r"./models/rfc_model.pkl", "wb") as fp:
        pickle.dump(cv_rfc, fp)
    with open(r"./models/logistic_model.pkl", "wb") as fp:
        pickle.dump(lrc, fp)



def main():
    '''
    Main call for all model build functionality
    input
        None

    output:
        None
    '''

    raw_data_path = r".\data\bank_data.csv"
    cat_recode_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    features_lst = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
         'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    print('main called')
    custs = import_data(raw_data_path)
    custs = derive_label(custs)
    perform_eda(custs)
    X_train, X_test, y_train, y_test = perform_feature_engineering(custs, cat_recode_lst, features_lst)
    train_models(X_train, X_test, y_train, y_test, features_lst)


if __name__ == "__main__":
    main()
