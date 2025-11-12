# Practical Application Assignment 17.1: Comparing Classifiers

**Contents**

 * [Introduction](#Introduction)
 * [How to use the files in this repository?](#how-to-use-the-files-in-this-repository)
 * [Business Understanding](#Business-Understanding)
 * [Data Understanding](#Data-Understanding)
 * [Data Preparation](#Data-Preparation)
 * [Baseline Model Comparison](#Baseline-Model-Comparison)
 * [Model Comparisons](#Model-Comparisons)
 * [Improving the Model](#Improving-the-Model)
 * [Next steps and Recommendations](#Next-steps-and-Recommendations)

 
## Introduction

This repository contains the Jupyter Notebook for the Application Assignment 17.1. This takes a sample jupyter notebook to complete the exercise to analyse UCI Bank Marketing Data Set in [bank-additional-full.csv](https://github.com/kondalraop/comparingclassifiers/blob/main/data/bank-additional-full.csv) file in the data folder of this repository to build a machine learning application that uses classifications to evaluate customers that will accept the Long-term deposit application using features like job, marital status, education, housing and personal loan.

The goal of this project is to compare the performance of the following classifiers namely 
* K Nearest Neighbor
* Logistic Regression
* Decision Trees and 
* Support Vector Machines

In comparing the models, the training times and accuracy of the models will be recorded. This should provide an indication on the model that will provide predictions to determine which customer will accept the long term deposit bank product via a phone based marketing campaign.

## How to use the files in this repository?

The notebooks are grouped into the following categories:
 * ``articles`` – More information on the data and features
 * ``data`` – bank-additional-full.csv data file from Kaggle Machine Learning dataset repository used in the notebooks
 * ``images`` – Image files used in Notebook
 * ``notebook`` – What Drives the customer subscribing long term deposit 


## Business Understanding

A bank aims to increase the number of customers subscribing to a term deposit (a fixed deposit investment) through targeted marketing campaigns. The bank runs several direct marketing campaigns, primarily through phone calls, to promote these term deposits.

Improve the efficiency and effectiveness of marketing campaigns by identifying customers most likely to subscribe to a term deposit.
Traditional mass marketing campaigns are expensive and inefficient — many customers contacted may not be interested, wasting time and resources.
Thus, the challenge is to predict which customers are most likely to accept the offer, so the bank can focus its marketing efforts effectively.

### Business Objective Definition

This dataset was provided for a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  The analysis of the data shows that the marketing campaign was not very successful in getting customers to sign up for the long term deposit product.

From a business objective,To predict whether a customer will subscribe to a term deposit (i.e., "yes" or "no" response) based on demographic, social, and campaign-related attributes.
  the task of this Machine Learning project is to determine which factors could lead to a higher success rate, for example,
- Identify characteristics of customers who are most likely to accept a term deposit offer.
- Help the marketing team plan more efficient campaigns by focusing resources on customers with higher likelihoods of acceptance.
- Reduce operational costs by minimizing unnecessary calls to customers unlikely to subscribe.
- Provide data-driven insights for strategic marketing decisions

## Data Understanding

To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the Materials and Methods section of the paper. How many marketing campaigns does this data represent?

According to the Materials and Methods section of the paper, the dataset was gathered from 17 marketing campaigns conducted between May 2008 and November 2010, comprising a total of 79,354 customer contacts.

In telephone marketing campaign, clients were offered an appealing long-term deposit product with competitive interest rates. Information from each contact—such as the client’s job, marital status, education level, and whether they held housing or personal loans—was recorded as part of the interaction.

The outcome of each contact was recorded as either a success or a failure, representing the target variable. Across the entire dataset, there were 6,499 successful subscriptions, corresponding to a success rate of approximately 8%.

Displayed below are some charts providing visualization on some of the observations of the dataset.


The first thing that was apparent from the provided data was that the low success rate of the marketing campaign in getting customers to sign up for the long term deposit product regardless of the features recorded for the customers (i.e., Education, Marital Status, job, contact etc.).

The one slight exception are customers with housing loan types where 52.4% signed up for the long term deposit product vs. 45.2% who did not.

An Alternative view on the data is to review number of succesful campaigns to see how features like education and job had a positive impact on the number of successful campaigns. See plots below:

Reviewing the plots where the customer signed up for the Bank Product/Marketing campaign was successful, you can observe the following:

- On Education, university degree folks said yes to the bank loan product
- For Job, bank had the most success with folks in admin role which is very broad, followed by Technician, then blue-collar


## Data Preparation

Apart from the imbalanced nature of the dataset, the following was done to prepare the dataset for modeling:
- Renamed "Y" feature to "subscribed" to make it more meaningful
- Use features 1 - 7 (i.e., job, marital, education, default, housing, loan and contact ) to create a feature set
- Use ColumnTransformer to selectively apply data preparation transforms, it allows you to apply a specific transform or sequence of transforms to just the numerical columns, and a separate sequence of transforms to just the categorical columns
- Use LabelEncoder to encode labels of the target column
- With your data prepared, split it into a train and test set. Next, we will split the data into a training set and a test set using the train_test_split function. We will use 30% of the data as the test set


## Baseline Model Comparison

For the baseline model, a DecisionTreeClassifier was selected. This algorithm is capable of performing multi-class classification and makes decisions by applying distinct feature subsets and decision rules at various stages of the classification process.

To evaluate performance, the Decision Tree model will be compared with a Logistic Regression model, which is commonly used to model the relationship between a dependent variable and one or more independent variables.

While Logistic Regression offers strong interpretability and performs well when sufficient data and domain understanding are available, the Decision Tree approach has certain limitations. In particular, it can be computationally intensive and sensitive to sample size, often requiring larger datasets to achieve stable and generalizable results.

In training, fitting and predicting both models on the dataset, the following results were observed:

| Model Name  	        | Accuracy                                | Precision	                    | Recall 	                    | F1_Score                    | Fit Time (ms) 
|-------------	        |:---------------------------------- 	|:-------------------------:	     |:----------------------:	|:----------------------:	|:----------------------:   |
| Decision Tree       	| 0.887513                              | 0.443792                  	     | 0.499954                    |  0.470202                   | 128                       |
| Logistic Regression   | 0.887594                               | 0.443797                     	| 0.500000                    |  0.470225                   | 193                       |
                        |

A preliminary review of the results indicates that the accuracy scores for both models were relatively high, exceeding 88%. However, the recall, precision, and F1-scores were all below 50%.

This discrepancy suggests that the classifier produced a substantial number of false negatives, meaning that many customers who would subscribe to a deposit were incorrectly predicted as non-subscribers. Such performance issues are likely attributable to class imbalance within the dataset, where the majority of records correspond to customers who did not subscribe to a term deposit (Deposit = "No"). Alternatively, suboptimal model performance could also result from untuned hyperparameters, but the imbalance in class distribution appears to be the more significant contributing factor.

## Model Comparisons

In this section, we will compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, fit and score each. Also, be sure to compare the fit time of each of the models.

| Model Name            | Train Time (s)               | Train Accuracy         | Test Accuracy 	     | 
|-------------------    |:---------------------------	|:---------------------:	|:----------------------:|
| Logistic Regression   | 0.398                        | 0.8872047448926502     | 0.8875940762320952     |  
| KNN                   | 72                           | 0.8911935069890049     | 0.8718944727684713     |  
| Decision Tree	        | 0.488                        | 0.8911935069890049     | 0.8846807477543093     |  
| SVM                   | 42.5	                       | 0.8873087995560335     | 0.8875131504410455

An examination of the model comparison results reveals that the Logistic Regression model outperformed the Decision Tree classifier across all evaluated metrics. It achieved the highest training and testing accuracy scores, demonstrated superior performance on precision, recall, and F1-score, and required the shortest training time in seconds, indicating both efficiency and stability in its predictive capability.

## Improving the Model

This dataset is so imbalanced when you look at the Exploratory section of this Notebook. Using these features to see if we can get a higher percentage of successful sign up for long term product did not provide a positive result with the exception of customer that have housing loan with a number of 52.4%

Using Grid Search to create models with the different parameters and evaluate the performance metrics

| Model Name        	| Train Time (s)               | Best Parameters                                                                         | Best Score 	                | 
|-------------------	|:---------------------------  |:-------------------------------------------------:	                                     |:----------------------:	 |
| Logistic Regression   | 85                            | 'model__C': np.float64(0.001), 'model__penalty': 'l1', 'model__solver': 'liblinear'	 | 0.8872394393842521           |  
| KNN                   | 387                           | 'model__n_neighbors': 17                                                               | 0.8864069611761135           |  
| Decision Tree         | 23.2                          | 'model__criterion': 'entropy', 'model__max_depth': 1, 'model__min_samples_leaf': 1     | 0.8872394393842521           |  
| SVM                   | 824                           | 'model__C': 0.1, 'model__kernel': 'rbf'                                                | 0.8872394393842521           |  


During the Support Vector Machine (SVM) experimentation, multiple parameter configurations were tested. Several runs were terminated prematurely due to excessive computation time, with some processes exceeding more than hour. The final configuration that successfully completed training utilized the following parameter grid:

 - param_grid_svc2 = { 'model__C': [ 0.1, 0.5, 1.0 ], 'model__kernel': ['rbf','linear'] }

This setup completed execution in approximately fourteen minutes.

An interesting observation emerged from the model comparison: the Logistic Regression, Decision Tree, and Support Vector Machine models each achieved similar best performance scores, despite utilizing distinct optimal parameter settings. In contrast, the K-Nearest Neighbors (KNN) model produced the lowest best score among the evaluated algorithms. Nevertheless, all models achieved relatively high accuracy levels, exceeding 85%.

## Next Steps and Recommendations

The main question that I have is the imbalanced dataset which is heavily weighted towards the unsuccessful marketing campaigns. If the model is used to determine features that are making the marketing campaign unsuccessful, then the models above could be useful.


Alternatively, the model can be used by the financial institution to understand customer profile that they need to target, for example, there was a high score amongst the "Yes" for customers contacted via Cellular so maybe the Bank can adopt modern features like Text Messages, Social Media platforms (i.e. Facebook, Instagram, Twitter, Tik Tok etc) for marketing campaigns

***License***

Open source projects are made available and contributed to under licenses that include terms that, for the protection of contributors, make clear that the projects are offered “as-is”, without warranty, and disclaiming liability for damages resulting from using the projects.
