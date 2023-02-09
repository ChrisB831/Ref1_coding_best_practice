# Summary 
This project focusses on 
* Testing
* Logging
* Best coding practicies
  * Running as a CLI
  * Running Pylint
  * Running autopep8



# Predict Customer Churn
Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
This project identifies credit card customers that are most likely to churn. The project contains a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also has the flexibility of being run interactively or from the command-line interface (CLI).

The data is available [**Here**](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code)



# Project Description
The project structure is 
```
<content_root>
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   └── results
├── logs
├── models
├── churn_library.py
├── churn_script_logging_and_tests.py
├── LICENSE
├── README.me
└── requirements.txt
```

where
| Name | Type | Description |
| :-- | :-: | :-- |
| data | directory | development data |
| images\eda | directory | EDA output |
| images\results | directory | model performance analysis output |
| logs | directory | test logs |
| models | directory | saved models |
| churn_library.py | Python module | model build module |
| churn_script_logging_and_tests.py | Python module |module build module test script|
| LICENSE | licence file|licence T&Cs|
| README.me | markdown file | this project readme |
| requirements.txt |text file | project dependencies |



# Files and data description



## `bank_data.csv`
This dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.
Churn rate is 16.07% of customers who have churned, there are no missing value in the dataset.
The first field contains the row index



## `churn_library.py`
This is the main model build module, containing all the code needed to:
* Import the data
* Derive the target field
* Perform an EDA and write the results to the `images\eda` folder
* Run feature engineering and generate the train and test datasets
* Build Logistic regression and a Decision tree (optimised over a hyperparameter search space) 
* Assess the performance of the model and write the result to the `images\results` folder
* Save (pickle) the most performant models to the `model` folder



The module can be executed as either as a top level script via
`python churn_library.py`
or via importing the module and running the `main` function


The `main` function contains three objects
| Name | Type | Description |
| :-- | :-: | :-- |
| raw_data_path | string | Path and name of the development data |
| cat_recode_list | list | Categorial feature names to recode |
| feature_list | list | Feature names to include in training |



## `churn_script_logging_and_tests.py`
This module contains all the code to test the main module build module. This focusses on the following criteria:
* Check the returned  items are not empty
* Check that folders are populated with required results 

The resulting test log (`churn_library_test.log`)  is written to the `logs` folder

The test script can be module executed as a top level script via
`python churn_script_logging_and_tests.py`



## `requirements.txt`
Contains all the dependencies for the project, these mat be installed via  
`python -m pip install -r requirements.txt`

