# Introduction

This directory is author's contribution to to support the 3W Project, which aims to promote experimentation and development of Machine Learning-based approaches and algorithms for specific problems related to detection and classification of undesirable events that occur in offshore oil wells.

The 3W Project is based on the 3W Dataset, a database described in this paper, and on the 3W Toolkit, a software package that promotes experimentation with the 3W Dataset for specific problems. 

# Table of contents

# Project description

## EDA

This project performs dynamic exploratory data analysis (EDA) based on the instances label of undesirable events. The EDA covers time series visualization, line graph for each of sensor variable, 
box plot of different classes. 

eg:

![image](https://github.com/user-attachments/assets/040ba106-f4ef-4955-8c4a-3db30426fafa)
![image](https://github.com/user-attachments/assets/4250c8bb-ac89-4cef-84e6-9bde4457c010)


## Modifying dataset
- Through careful analysis, suitable datasets from the dataset directory were selected, resampled (by removing missing data, data noise removal through moving average, and z-score outlier removals to prepare for ML modeling and validation. These resampled dataset were saved in the trainDataset directory.
- To capture temporal dependencies, time windowing is implemented onto the resampled instances, generating up to 15 samples with 60 observations each.
- Feature extraction is done by using the tsfresh librart to extract statistical features including mean, median, standard deviation, variance, maximum and minimum values in each generaated time window for each of the sensor data.
- Since this project to target anomalies, the normal class label is replaced with 0 to indicate negative anomaly, while transient faulty and faulty classes are identified as 1; indicating positive anomaly.
- SMOTE was implemented to generate synthetic samples for the minority classes.

## Modeling

- Random Forest
- XGBoost
- Local Outlier Factor

Models were optimized by using GridSearch cross validation.

## Assess
By using the test datasets, models were assessed and validated through several metrics including:
- Accuracy
- Precision
- F1-score
- AUPRC
- AUROC

## Implentation
Through assessment and validation, the best model were identified as Random Forest. Explainable AI was added in the implementation to provide transparenct to the model. SHAP tree explainer was implemented to generate a SHAP summary plot.

# Reproducibility
The eda.ipynb and the model.ipynb are made to run based on prompted undesirable event label (0-8),and can be reproduced independently of each other. 

To initialize a local Jupyter Notebook server:

* To initialize a local Jupyter Notebook server:
```
$ jupyter notebook
```

* To run the streamlit app locally:
```
$ streamlit run ain/app.py
```
