# Heart-Disease-Prediction-ML_
This project focuses on predicting whether a person has heart disease or not using machine learning techniques. The dataset used is heart_2020_cleaned from Kaggle, which includes various health parameters such as age, BMI, sleep hours, smoking habits, and more.
## Project Overview

The goal of this project is to build a predictive model that can accurately identify individuals with heart disease, prioritizing recall to minimize false negatives. The project involves exploratory data analysis (EDA), data preprocessing, model training, and evaluation.

## Dataset

The dataset contains the following key features:
- **Demographics**: Age, Sex
- **Health Metrics**: BMI, General Health, Sleep Hours
- **Medical History**: Diabetes, Past Heart Stroke
- **Lifestyle Factors**: Smoking, Alcohol Consumption

## Key Observations from EDA

1. **BMI**: Obesity is linked to a higher likelihood of heart disease.
2. **Age**: Older individuals have a significantly higher chance of heart disease.
3. **Sex**: Men are more likely to have heart disease compared to women.
4. **Sleep Hours**: Both insufficient (0-4 hrs) and excessive (10+ hrs) sleep are associated with higher heart disease rates.
5. **Diabetes**: People with diabetes have a higher chance of heart disease.
6. **Smoking**: Smokers are twice as likely to develop heart disease.
7. **Alcohol**: The dataset lacked detailed alcohol consumption data, making conclusions inconclusive.

## Data Preprocessing

- **Encoding**: Categorical features were converted using Ordinal Encoding for ordinal variables and One-Hot Encoding for nominal variables.
- **Class Imbalance**: Addressed using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.

## Models and Hyperparameter Tuning

### Logistic Regression
- **Class Weight**: Used `class_weight='balanced'` to handle imbalanced data.
- **Regularization**: Optimal performance with L1 regularization (Lasso) and `C=0.0001`.
- **Max Iterations**: Set to `10` as the model converged early.

### Decision Trees
- **Max Depth**: Optimal depth found to be `5`, balancing complexity and generalization.

### KNN Classifier
- **Neighbors**: Best recall achieved with `n_neighbors=15`.

## Evaluation Metrics
- **Recall**: Prioritized to minimize false negatives.
- **Precision-Recall Curve**: Used for evaluation due to class imbalance, outperforming a random classifier.

## Challenges and Learnings
- **Class Imbalance**: Addressed using SMOTE to improve model performance.
- **Feature Correlation**: Low correlation between features and heart disease, with age showing the highest correlation (0.23).
- **Model Limitations**: K-means++ performed poorly due to its clustering nature and inability to handle class imbalance.

## Repository Structure
- `data/`: Contains the dataset (`heart_2020_cleaned.csv`).
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, and model training.
- `src/`: Source code for the project, including utility functions and model scripts.
- `results/`: Outputs such as graphs, metrics, and model evaluations.

## Results
- The best-performing model achieved a recall of 0.82 using Decision Trees, significantly outperforming a random classifier. 

