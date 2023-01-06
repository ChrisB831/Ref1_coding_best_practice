'''
TBD

Author: Chris Bonham
Date: 5th January 2023
'''
import logging
import os
import pandas as pd
import churn_model_build as cl


logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')



def test_import(import_data):
    '''Desc
    Input:
        inputs
    Output:
        outputs
    '''
    try:
        # Test: Does data import?
        df = import_data(r".\data\bank_data.csv")
        logging.info("test import_data: SUCCESS. File found")

        # Test: After successful import, does DF contain rows and columns?
        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            logging.info("test import_data: SUCCESS. Shape of dataframe is %s", df.shape)

        except AssertionError:
            logging.error("test import_data: FAILURE. Dataset has no rows or columns")

    except FileNotFoundError:
        logging.error("test import_data: FAILURE. File not found")

    return df


def test_derive_label(derive_label, df):
    '''Desc
    Input:
        inputs
    Output:
        outputs
    '''
    try:
        # Test: oes a column called 'Churn' exist on dataframe
        derive_label(df)
        assert 'Churn' in df.columns.values
        logging.info("test derive_label: SUCCESS. 'Churn' field has been created")

    except AssertionError:
        logging.error("test derive_label: FAILURE. 'Churn' field has not been created")

    return df



def count_dir_files(pth):
    ''''Helper function to count the number of files in pth
    input:
        pth: String of directory path
    output:
        Integer count of the number of files in pth
    '''
    count = 0
    for blob in os.listdir(pth):
        # check if current path is a file
        if os.path.isfile(os.path.join(pth, blob)):
            count += 1
    return count



def test_eda(perform_eda, df):
    '''Desc
    Input:
        inputs
    Output:
        outputs
    '''
    try:
        # Test: Does the output directory exist
        assert os.path.exists(r".\images\eda")
        logging.info("test eda: SUCCESS. Output path exists")

        try:
            # Test: Does perform_eda generate the correct number of files?
            tot_files = len(df.select_dtypes(include='number').columns.tolist()) + \
                        len(df.select_dtypes(include=['object', 'category']).columns.tolist())

            perform_eda(df)
            assert count_dir_files(r".\images\eda") == tot_files
            logging.info("test eda: SUCCESS. Correct number of eda files generated")

        except AssertionError:
            logging.error("test eda: FAILURE. Incorrect number of eda files generated")

    except AssertionError:
        logging.error("test eda: FAILURE. Output path does not exist")



def test_perform_feature_engineering(perform_feature_engineering,
                                     df,
                                     cat_rec_lst,
                                     feat_lst):
    '''Desc
    Input:
        inputs
    Output:
        outputs
    '''
    try:
        Xtrain, Xtest, ytrain, ytest = perform_feature_engineering(
            df,
            cat_rec_lst,
            feat_lst)

        # Test: Have the numpy arrays been created
        assert isinstance(Xtrain,pd.DataFrame)
        assert isinstance(Xtest,pd.DataFrame)
        assert isinstance(ytrain,pd.Series)
        assert isinstance(ytest,pd.Series)

        logging.info("test perform_feature_engineering SUCCESS. Training data has been created")

    except AssertionError:
        logging.error("test perform_feature_engineering FAILURE. Training data is not complete")

    return Xtrain, Xtest, ytrain, ytest


def test_train_models(train_models, Xtrain, Xtest, ytrain, ytest, feat_lst):
    '''Desc
    Input:
        inputs
    Output:
        outputs
    '''
    try:
        train_models(Xtrain, Xtest, ytrain, ytest, feat_lst)

        # Test: Have the training results been created
        assert os.path.exists(r".\images\results\classification_report.txt")
        assert os.path.exists(r".\images\results\RF feature importance.png")
        assert os.path.exists(r".\images\results\Test ROC curve.png")
        assert os.path.exists(r".\images\results\Train ROC curve.png")

        try:
            # Test: If created test they are not empty
            assert os.stat(r".\images\results\classification_report.txt").st_size != 0
            assert os.stat(r".\images\results\RF feature importance.png").st_size != 0
            assert os.stat(r".\images\results\Test ROC curve.png").st_size != 0
            assert os.stat(r".\images\results\Train ROC curve.png").st_size != 0
            logging.info("test train_model SUCCESS. Training results created")

        except AssertionError:
            logging.error("test train_model FAILURE. One or more of training results file are "
						  "empty")

    except AssertionError:
        logging.error("test train_model FAILURE. One or more of training results file are "
					  "missing")

    try:
    # Test: Have the models files been created
        assert os.path.exists(r".\models\logistic_model.pkl")
        assert os.path.exists(r".\models\rfc_model.pkl")

        try:
            # Test: If created test they are not empty
            assert os.stat(r".\models\logistic_model.pkl").st_size != 0
            assert os.stat(r".\models\rfc_model.pkl").st_size != 0
            logging.info("test train_model SUCCESS. Models saved")

        except AssertionError:
            logging.error("test train_model FAILURE. One or more model files are empty")

    except AssertionError:
        logging.error("test train_model FAILURE. One or more model files are missing")



if __name__ == "__main__":

    # Define parameters
    cat_recode_lst = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'
    ]
    features_lst = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    custs = test_import(cl.import_data)
    custs = test_derive_label(cl.derive_label, custs)
    test_eda(cl.perform_eda, custs)
    X_train, X_test, y_train, y_test = \
		test_perform_feature_engineering(cl.perform_feature_engineering,
                                         custs,
                                         cat_recode_lst,
                                         features_lst)
    test_train_models(cl.train_models, X_train, X_test, y_train, y_test, features_lst)
