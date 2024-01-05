# The Machine Learning Project deployment using Flask and Docker


## Name: Student Performance Prediction 
<img src="https://github.com/shreyaskorde16/mlproject/blob/main/Screenshot%202024-01-05%20230547.png" width="1100" height= "700" />


<p align="justify">
This project aims to find the best model for the dataset base on r2_score among Linear Regression, Random Forest Regressor, Decision Tree Regressor, Gradient Bosting regressor, K-Neighbor Regressor, XGBRegressor, Cat Boost regressor, AdaBoost regressor using Hyperparameter tunning and GridsearcCV.

<p align="justify">
 
### Life cycle of Machine learning Project
 + **Problem statement**: This project investigates how additional variables such as gender, ethnicity, parental level of education, lunch, and test preparation course affect student performance (test scores).
   
 + **Data collection**: Dataset source https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977 This data set consists of the marks secured by the students in various subjects. Inspiration is understanding the influence of the parent's background, test preparation, etc on students' performance.

 + **Dataset Information**:
    - gender: sex of students -> (Male/female)
    - race/ethnicity: ethnicity of students -> (Group A, B, C, D, E)
    - parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
    - lunch: having lunch before the test (standard or free/reduced)
    - test preparation course: complete or not complete before the test
    - math score
    - reading score
    - writing score

---

+ **Repository Structure**:

`src/logger.py`: scripts for logging Information

`src/exception.py`: for handline exception

`src/utils.py`: To load and save the .pkl file

`src/components/data_ingesion.py`: For loading the dataset and converting train and test dataset

`src/components/data_transformation.py`: To transform and preprocess data using ML Pipeline and ColumnTransformer

`src/components/model_trainer.py`: To train and find the best model suited for the dataset

`app.py`: File contains Flask web app code

`Docker`: Docker file to build docker image name as 
[shreyaskorde16/student_performance](https://hub.docker.com/r/shreyaskorde16/student_performance/tags)


