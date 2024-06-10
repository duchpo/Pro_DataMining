# Table of Contents
# House Price Prediction | Machine Learning  Final Project
---


![](https://s3.eu-north-1.amazonaws.com/ammar-files/kaggle-kernels/House+Price+Prediction+%7C+An+End-to-End+Machine+Learning+Project/header-img.jpg)

<p>&nbsp;</p>

# Các Phần Chính

<a href="#introduction">Giới Thiệu</a>

<a href="#data-prep">Chuẩn bị dữ Liệu </a>

<a href="#eda">Exploratory Data Analysis</a>

<a href="#pred-type">Các Mô Hình Sử Dụng</a>

<a href="#model-building">Xây Dựng Và Đánh Giá Mô Hình </a>

<a href="#analysis-comparison">Phân Tích và So Sánh</a>

<a href="#comparison">Kết Luận</a>


---

## General Information
Referencing to the published project on Rpubs: [diabetes-analyzing-ml](https://rpubs.com/yourlink/diabetes-analyzing-ml).

Overall of dataset: women's medical and demographic data to predict diabetes. This dataset contains information on 769 women and includes many health-related attributes. Here is a brief overview of the columns:

- **Pregnancy**: The number of times a woman has been pregnant.
- **Glucose**: The concentration of glucose in a woman's plasma.
- **Blood pressure**: Measure of blood pressure.
- **Skin thickness**: The thickness of the skin folds in the triceps.
- **Insulin**: Insulin concentration in the blood.
- **BMI (Body Mass Index)**: A measure of body fat based on height and weight.
- **Diabetes pedigree function**: A function that shows the likelihood of developing diabetes based on family history.
- **Age**: Age of the woman.
- **Outcome**: The target variable indicates whether the woman has diabetes (1 for diabetics, 0 for non-diabetics).

---

## Problem Solving

### 👨‍🏫 Exploring the Dataset and Pre-processing
- Describing the most overall vision for readers to comprehend what exactly this dataset's structure is.
- Utilizing some legible visualization techniques for plotting out the significant features of the dataset.
- Identifying any abnormal things in the dataset, such as null/nan data points or outliers, which will incorrectly affect the analysis process.

### 📊 Establishing the Prediction Model with Logistic Regression and Decision Tree
- This problem aims to forecast whether the patient has diabetes or not by analyzing the feature attributes, which have strong correlations with the Outcome variables.
- Observing the dataset to define which attributes are not necessary for these problems. Then, we will remove them before constructing the machine learning models.
- Comparing the performance and accuracy of the two models and concluding which one is better.

### 🗂 Classifying the Categories of Mass using Random Forest Model
- The problem serves for identifying the mass situation of the patient, such as underweight, normal, overweight, and obese. It will be helpful for doctors to keep track of the health of patients having a probability of diabetes.
- Observing the dataset to define which attributes are not necessary for these problems. Then, we will remove them before constructing the models.
- Performing fine-tuning tasks to select the best parameter values. Then, we can build the best possible model based on these fine-tuned parameters.

### 🕵️‍♀️ Hypothesis Validation using T-Test Technique
- Using One-sample T-test, we hypothesize that an average BMI (Body Mass Index) of 34 is susceptible to diabetes.
- Using Independent Samples T-test, we hypothesize that body fat (BMI) does not affect whether or not there is disease.
- Using One-sample T-test, we hypothesize that age also affects whether a person has diabetes.

---

## Technology
- **Environment**: RStudio, R interpreter.
- **Display mode**: R-Markdown or R-Notebook.
- **Packages**:
  - `glm` for logistic regression.
  - `rpart` for decision tree model.
  - `randomForest` for random forest models.

---

