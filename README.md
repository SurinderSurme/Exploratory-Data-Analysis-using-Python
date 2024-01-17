# Estimation of Obesity Level based on eating Habits and physical condition.


## Project Introduction

This report examines an obesity dataset and its associated variables. The dataset includes attributes such as frequency of high-calorie food consumption (FAVC), frequency of vegetable consumption (FCVC), number of main meals (NCP), consumption of food between meals (CAEC), daily water consumption (CH20), alcohol consumption (CALC), calories consumption monitoring (SCC), physical activity frequency (FAF), time spent using technology devices (TUE), mode of transportation used (MTRANS), gender, age, height, and weight. The dataset was labelled, and a NObesity class variable with categories like Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III was established.
The primary goal of this investigation is to investigate the link between various variables and the target variable NObesity, as well as to develop prediction models to identify the obesity type based on the supplied data.
This data was used to estimate obesity levels in adults aged 14 to 61 from Mexico, Peru, and Colombia, with a variety of eating habits and physical conditions. The information was analyzed after the data was obtained utilizing an online platform with a survey in which anonymous people answered each question.
## The following characteristics are associated with eating habits:
•	Frequent intake of high caloric meals (FAVC)
•	Frequent consumption of vegetables (FCVC)
•	Number of primary meals (NCP)
•	Consumption of food between meals (CAEC)
•	Daily water consumption (CH20)
•	Alcohol consumption (CALC).
•	Calories consumption monitoring (SCC)
•	Physical activity frequency (FAF)
•	Time utilizing technology devices.
•	(TUE) Transportation used.
•	(MTRANS) are the physical condition parameters.
•	Other characteristics retrieved were Gender, Age, Height, and Weight.
Finally, all data was labelled, and the class variable NObesity was formed using the BMI values shown below:
•	Underweight less than 18.5 lbs.
•	Normal 18.5 to 24.9 lbs.
•	Overweight 25.0 to 29.9 lbs.
•	Obesity I 30.0 to 34.9 lbs.
•	Obesity II 35.0 to 39.9 lbs.
•	Obesity III greater than 40 lbs.
•	Because the data contains both numerical and continuous information, it may be analyzed using algorithms for classification, prediction, segmentation, and association.
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/5506daf1-098a-4c32-b62f-67f526301ec6)


## Dataset Analysis
Handling missing numbers and outliers was part of the data wrangling process. The dropna() method was used to remove missing values before analyzing the remaining data. The IQR approach was used to find and eliminate outliers in the 'Height' and 'Weight' columns.



## EDA
EDA was performed to gain insights into the distribution and relationships between variables. Histograms, scatter plots, box plots, and count plots were used to visualize the data.
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/4575a912-c13d-4c12-aa47-f2dc2b49491d)


 
This graph shows that all of the output categories were given identical weightage while there is no data collection inconsistency in the data
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/74fc0953-af02-4d3c-aaa2-f3f07175e3eb)


 
This graph depicts the relationship between all of the numerical variables. As we can see, there is no multicollinearity in the data since the characteristics are not intercorrelated.

![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/27e79a75-4053-435e-ab43-0361bc816fae)

 

This graph depicts the height distribution in the data. The average height spans from 1.5 to 1.9 metres, with the majority of persons standing between 1.7 and 1.75 metres tall.

1.	Height Vs Nobeyesdad :
2.	![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/2db45ed4-7ea2-4a5d-a097-c69fe4abeb5c)
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/553978ce-c77e-4853-8db4-a19a67b53b44)

According to this graph, the distribution of height is even across all obesity groups.

3.	Weight Vs NObeysdad:  
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/01daa110-2c07-4329-b15c-23854639bbd1)

Weight distribution is unequal between obesity categories, which is noticeable since even little weight changes impact obesity. It also displays the weight ranges for each group.

 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/b8e48fda-af5a-4281-9c5e-4e78534857ad)

This graph depicts the height vs. weight dispersion. There is a thick dispersion where the height is 1.5 to 1.7 and the weight is 60 to 80.
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/3bc70bcd-aff6-4869-b7d6-8e460bf6bcc7)

This shows the distribution of age over the obesity type. From this we can also see the age group of the people lying in each category of obesity. We can also see some outliers for age of people who fall under Normal weight category

![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/331914f9-f1ab-4692-8ed6-f319e910f3a6)
 
This graphic shows that the average age of persons classified as Overweight_Level_II and Obsity_Type_I is older than that of those classified as other types of obesity. As a result, we may conclude that Overweight_Level_II and Obesity_Type_I are not more likely to occur in persons under the age of 20.
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/92e6f789-60db-4c5c-8ca3-4ead18906a58)

This figure shows that the density is highest when the age is between 15 and 25 and the weight is between 40 and 80.

This figure shows that consumption of water is almost same in each type except the normal weight
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/4c513645-efe5-4f3f-b24e-ed97e58e20bb)


 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/2bf665da-14cd-4760-b355-e35ca8b5755a)


![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/8474c985-a3e2-4276-bb87-04f8de5cbabb)


From this graph we can see that there is a bias in sampling of the data in terms of family history because we can see only a few samples of data for “no” category and we can see more number of data for yes category

![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/d4b83d6c-5e86-4c05-b929-2e7bc20994d1)

![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/b41829c3-6ee0-41f7-b331-cb88097b97cc)

![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/50a84977-56f9-490e-b667-cb7ce7da710b)


 
We can also observe a bias in the data gathering from this graph since we don't have enough data for SMOKE = "yes."

 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/881dd83e-0347-4d02-8849-7f2609a8eb08)
The graph demonstrates that those who consume high calorie foods outnumber those who do not.
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/47c2e02e-19e3-4c9a-aa25-60e6cf91a3a2)

This graph shows that people who consume food between meals sometimes are more compared to other categories.
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/49741f20-c091-491a-8158-096aa7c7dbd0)

The plot shows that people who consume food between meals sometimes are most likely to be in the age of 20-40
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/93295c1b-fbc8-4832-ba2a-111c44775ba8)

This graph shows that consumption of alcohol sometimes is more while compared to other categories.
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/1ece3039-0f57-4ecb-a97b-7d1a09aca6c0)

This box plot demonstrates that most alcoholic drinkers are between the ages of 20 and 30.
 
 ![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/b9c2340b-a377-4b28-b44f-cd2b6b40eb6c)

The graph shows that both Gender consumes equal amount of caloric food as they age






Out[131]:
	Gender	Age	Height	Weight	family_history_with_overweight	FAVC	FCVC	NCP	CAEC	SMOKE	CH2O	SCC	FAF	TUE	CALC	MTRANS	NObeyesdad
0	0	21.0	1.62	64.0	1	0	2.0	3.0	1	0	2.0	0	0.0	1.0	0	0	Normal_Weight
1	0	21.0	1.52	56.0	1	0	3.0	3.0	1	1	3.0	1	3.0	0.0	1	0	Normal_Weight
2	1	23.0	1.80	77.0	1	0	2.0	3.0	1	0	2.0	0	2.0	1.0	2	0	Normal_Weight
3	1	27.0	1.80	87.0	0	0	3.0	3.0	1	0	2.0	0	2.0	0.0	2	1	Overweight_Level_I
4	1	22.0	1.78	89.8	0	0	2.0	1.0	1	0	2.0	0	0.0	0.0	1	0	Overweight_Level_II
As we can see, each string values have been convert into numeric value according to their 
We can observe that each variable is globally linearly independent of the others. As a result, ACP will be ineffective, hence we shall not conduct it. We will choose the variable that is associated with less than 30%.
![image](https://github.com/SurinderSurme/Exploratory-Data-Analysis-using-Python/assets/103140222/3ff81928-ed5e-483c-ac28-3886c6c637ca)


Key Findings from EVA:
The dataset has a balanced representation of weight categories, which provides enough data for analysis. There are no significant correlations between the numerical variables, indicating that there are no difficulties with multicollinearity.
Feature Choice
Building an effective machine learning model requires careful feature selection. Two approaches of feature selection were used:
Variance Threshold: To minimise dimensionality and enhance model performance, features with low variance (below the set threshold) were deleted.
Select-K-Best: Based on the F-regression score, the top K features with the best predictive power were chosen. The selected characteristics were utilised for modelling after being picked using feature selection techniques.
Model Implementation
Several machine learning models were considered for predicting the weight category based on the selected features. The models used are as follows:

Logistic Regression:
A simple yet effective classification algorithm suitable for binary and multiclass problems.
Model summary:
precision    recall  f1-score   support

Insufficient_Weight       0.81      0.86      0.84       107
Normal_Weight       0.63      0.51      0.56        94
Obesity_Type_I       0.81      0.83      0.82       109
Obesity_Type_II       0.92      0.96      0.94       108
Obesity_Type_III       1.00      0.97      0.99       102
Overweight_Level_I       0.61      0.65      0.63        84
Overweight_Level_II       0.65      0.65      0.65        92

accuracy                           0.79       696
macro avg       0.78      0.78      0.78       696
weighted avg       0.79      0.79      0.79       696
Logistic regression model is trained for the following c values and the optimum model is obtained C : Regularization parameter. 1,5,10,20

Accuracy: 0.7887931034482759

Gradient Boosting Classifier: 
An ensemble technique combining weak learners (decision trees) to create a strong classifier.
Append the model's performance metrics (accuracy and model name "Gradient Boosting Classifier") to the 'perf' list:
perf.append([score, "Gradient Boosting Classifier"])
Confusion matrix:
[[ 98   6   0   1   0   1   1]
[  1  77   4   1   1   9   1]
[  0   2 105   1   0   0   1]
[  0   3   2 103   0   0   0]
[  1   1   0   2  97   0   1]
[  0   8   1   0   0  72   3]
[  0   6   5   0   0   2  79]]
Model summary:
precision    recall  f1-score   support

Insufficient_Weight       0.98      0.92      0.95       107
Normal_Weight       0.75      0.82      0.78        94
Obesity_Type_I       0.90      0.96      0.93       109
Obesity_Type_II       0.95      0.95      0.95       108
Obesity_Type_III       0.99      0.95      0.97       102
Overweight_Level_I       0.86      0.86      0.86        84
Overweight_Level_II       0.92      0.86      0.89        92

accuracy                           0.91       696
macro avg       0.91      0.90      0.90       696
weighted avg       0.91      0.91      0.91       696

Accuracy: 0.9066091954022989


Support Vector Machine (SVM): A powerful algorithm for classification tasks, effective in handling high-dimensional data.
Append the model's performance metrics (accuracy and model name "SVM2") to the 'perf' list
perf.append([score, "SVM2"])
Confusion matrix:
[[ 97   7   0   1   0   1   1]
[ 11  68   3   1   0  10   1]
[  0   1 101   0   0   1   6]
[  0   3   2 103   0   0   0]
[  2   0   0   0  99   0   1]
[  0   8   1   0   0  66   9]
[  0   4   3   1   0   5  79]]
Model summary:
precision    recall  f1-score   support

Insufficient_Weight       0.88      0.91      0.89       107
Normal_Weight       0.75      0.72      0.74        94
Obesity_Type_I       0.92      0.93      0.92       109
Obesity_Type_II       0.97      0.95      0.96       108
Obesity_Type_III       1.00      0.97      0.99       102
Overweight_Level_I       0.80      0.79      0.79        84
Overweight_Level_II       0.81      0.86      0.84        92

accuracy                           0.88       696
macro avg       0.88      0.88      0.88       696
weighted avg       0.88      0.88      0.88       696

Accuracy: 0.8807471264367817


Naive Bayes: A probabilistic classifier based on Bayes' theorem, suitable for handling discrete data.
Append the model's performance metrics (accuracy and model name "Naive Bayes") to the 'perf' list
perf.append([score, "Naive Bayes"])
Confusion matrix:
[[87 15  0  1  0  3  1]
[43 24  8  0  0 11  8]
[ 0  2 60 41  3  2  1]
[ 0  1  3 99  0  2  3]
[ 2  2  0  0 98  0  0]
[ 2 10 31  1  1 33  6]
[ 0  6 43 11  1  2 29]]
Model summary:
precision    recall  f1-score   support

Insufficient_Weight       0.65      0.81      0.72       107
Normal_Weight       0.40      0.26      0.31        94
Obesity_Type_I       0.41      0.55      0.47       109
Obesity_Type_II       0.65      0.92      0.76       108
Obesity_Type_III       0.95      0.96      0.96       102
Overweight_Level_I       0.62      0.39      0.48        84
Overweight_Level_II       0.60      0.32      0.41        92

accuracy                           0.62       696
macro avg       0.61      0.60      0.59       696
weighted avg       0.61      0.62      0.60       696

Accuracy: 0.617816091954023

Random Forest: A popular ensemble method using multiple decision trees for classification tasks.
confusion matrix
[[ 94  10   0   1   0   1   1]
[  4  79   3   0   1   6   1]
[  0   1 106   0   0   0   2]
[  0   3   1 104   0   0   0]
[  1   1   0   0  99   1   0]
[  0   7   1   0   0  73   3]
[  0   4   3   0   0   1  84]]
model summary :
precision    recall  f1-score   support

Insufficient_Weight       0.95      0.88      0.91       107
Normal_Weight       0.75      0.84      0.79        94
Obesity_Type_I       0.93      0.97      0.95       109
Obesity_Type_II       0.99      0.96      0.98       108
Obesity_Type_III       0.99      0.97      0.98       102
Overweight_Level_I       0.89      0.87      0.88        84
Overweight_Level_II       0.92      0.91      0.92        92

accuracy                           0.92       696
macro avg       0.92      0.92      0.92       696
weighted avg       0.92      0.92      0.92       696

accuracy : 0.9181034482758621
The models were trained on the training data and evaluated on the test data using accuracy, precision, recall, and F1-score metrics.

Results Interpretation and Implications
The machine learning models were evaluated based on their performance metrics. The results are summarized below:
Random Forest Classifier:
Accuracy: 91.81%
Precision, Recall, and F1-score: The model achieved high precision, recall, and F1-score for most classes, indicating good performance in predicting different weight categories. The model is especially effective for Obesity Type II and Obesity Type III.
Implications: Random Forest is a powerful ensemble method that performs well in both binary and multiclass classification tasks. Its ability to handle complex relationships in data and reduce overfitting makes it a reliable choice for this problem.
Gradient Boosting Classifier:

Accuracy: 90.66%
Precision, Recall, and F1-score: The model shows strong performance in predicting most weight categories, especially for Obesity Type I, Obesity Type II, and Obesity Type III.
Implications: Gradient Boosting is another ensemble technique that can achieve high accuracy by combining multiple weak learners. It is an effective choice for multiclass classification and can handle complex data distributions.
Support Vector Machine (SVM):

Accuracy: 88.07%
Precision, Recall, and F1-score: SVM performed well for most classes, particularly for Insufficient Weight, Obesity Type I, and Obesity Type II.
Implications: SVM is a powerful algorithm for classification tasks, especially when dealing with high-dimensional data. It is a versatile choice, but hyperparameter tuning and data scaling can significantly impact its performance.
Naive Bayes:

Accuracy: 61.78%
Precision, Recall, and F1-score: Naive Bayes achieved moderate performance for some classes, but struggled with others. It performed best for Obesity Type II and Obesity Type III.
Implications: Naive Bayes is a simple and fast classifier, suitable for discrete data. While it may not perform as well as other algorithms on this specific dataset, it can be useful for certain types of problems.
Overall, the Random Forest Classifier stands out with the highest accuracy and balanced performance across all weight categories. It is the recommended model for predicting weight categories based on the provided dataset. However, further exploration and experimentation with other algorithms, hyperparameter tuning, and feature engineering may lead to even better results in the future. It's essential to continue refining the model to improve its performance and generalization capabilities on new data.






Out-of-Sample Predictions
Out-of-sample predictions were performed to simulate the model's performance in a real-world production environment using new data. This data was obtained separately from the test dataset used during model evaluation. The model's predictions were analyzed and compared to the actual weight categories to assess its performance in real-world scenarios.

Concluding Remarks
In conclusion, the analysis of the Obesity dataset has provided valuable insights into the factors influencing weight categories. The machine learning models developed in this project can effectively predict an individual's weight category based on their dietary and physical habits.

It is essential to note that the choice of the most suitable model depends on the specific objectives and requirements of the project. Further analysis and fine-tuning of the models may be necessary to optimize their performance for specific use cases.

Overall, this project contributes to the understanding of obesity-related factors and offers predictive models that can assist in healthcare decision-making and interventions.


import matplotlib.pyplot as plt
import seaborn as sns
import datetime
Reading and manipulating data
df_raw = pd.read_csv("Train.csv")
Previewing data and datatypes
df_raw.head()

fecha_dato	ncodpers	ind_empleado	pais_residencia	sexo	age	fecha_alta	ind_nuevo	antiguedad	indrel	...	ind_hip_fin_ult1	ind_plan_fin_ult1	ind_pres_fin_ult1	ind_reca_fin_ult1	ind_tjcr_fin_ult1	ind_valo_fin_ult1	ind_viv_fin_ult1	ind_nomina_ult1	ind_nom_pens_ult1	ind_recibo_ult1
0	2015-01-28	1375586	N	ES	H	35	2015-01-12	0.0	6	1.0	...	0	0	0	0	0	0	0	0.0	0.0	0
1	2015-01-28	1050611	N	ES	V	23	2012-08-10	0.0	35	1.0	...	0	0	0	0	0	0	0	0.0	0.0	0
2	2015-01-28	1050612	N	ES	V	23	2012-08-10	0.0	35	1.0	...	0	0	0	0	0	0	0	0.0	0.0	0
3	2015-01-28	1050613	N	ES	H	22	2012-08-10	0.0	35	1.0	...	0	0	0	0	0	0	0	0.0	0.0	0
4	2015-01-28	1050614	N	ES	V	23	2012-08-10	0.0	35	1.0	...	0	0	0	0	0	0	0	0.0	0.0	0
5 rows × 48 columns
Renaming Columns
#renaming columns

dict = {'fecha_dato' : 'Date',
       'ncodpers' : 'Customer_Code',
       'ind_empleado' : 'Employee_Index',
       'pais_residencia' : 'Country',
       'sexo' : 'Gender',
       'age' : 'Age',
       'fecha_alta' : 'Customer_Join_Date',
       'ind_nuevo' : 'Customer_Index',
       'antiguedad' : 'Customer_Seniority',
       'indrel' : 'Primary_Customer',
       'ult_fec_cli_1t' : 'Customer_Leave_Date',
       'indrel_1mes' : 'Customer_Type',
       'tiprel_1mes' : 'Customer_Relation',
       'indresi' : 'Residence_Index',
       'indext' : 'Foriegner_Index',
       'conyuemp' : 'Spouse_Index',
       'canal_entrada' : 'Channel_Used',
       'indfall' : 'Deceased_Index',
       'tipodom' : 'Primary_Address',
       'cod_prov' : 'Customer_Address',
       'nomprov' : 'Province',
       'ind_actividad_cliente' : 'Activity_Index',
       'renta' : 'Gross_Income',
       'segmento' : 'Segmentation',
       'ind_ahor_fin_ult1' : 'Saving_Account',
        'ind_aval_fin_ult1' : 'Guarantees',
        'ind_cco_fin_ult1' : 'Current_Accounts',
        'ind_cder_fin_ult1' : 'Derivative_Account',
        'ind_cno_fin_ult1' : 'Payroll_Account',
        'ind_ctju_fin_ult1' : 'Junior_Account',
        'ind_ctma_fin_ult1' : 'More_Private_Account',
        'ind_ctop_fin_ult1' : 'Private_Account',
        'ind_ctpp_fin_ult1' : 'Private_Plus_Account',
        'ind_deco_fin_ult1' : 'Short_Term_Deposits',
        'ind_deme_fin_ult1' : 'Medium_Term_Deposits',
        'ind_dela_fin_ult1' : 'Long_Term_Deposits',
        'ind_ecue_fin_ult1' : 'E_Account',
        'ind_fond_fin_ult1' : 'Funds',
        'ind_hip_fin_ult1' : 'Mortgage',
        'ind_plan_fin_ult1' : 'Pensions',
        'ind_pres_fin_ult1' : 'Loans',
        'ind_reca_fin_ult1' : 'Taxes',
        'ind_tjcr_fin_ult1' : 'Credit_Card',
        'ind_valo_fin_ult1' : 'Securities',
        'ind_viv_fin_ult1' : 'Home_Account',
        'ind_nomina_ult1' : 'Payroll',
        'ind_nom_pens_ult1' : 'Pensions2',
        'ind_recibo_ult1' : 'Direct_Debit'
       }

df_raw = df_raw.rename(columns = dict)
datatypes
df_raw.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13647309 entries, 0 to 13647308
Data columns (total 48 columns):
 #   Column                Dtype  
---  ------                -----  
 0   Date                  object 
 1   Customer_Code         int64  
 2   Employee_Index        object 
 3   Country               object 
 4   Gender                object 
 5   Age                   object 
 6   Customer_Join_Date    object 
 7   Customer_Index        float64
 8   Customer_Seniority    object 
 9   Primary_Customer      float64
 10  Customer_Leave_Date   object 
 11  Customer_Type         object 
 12  Customer_Relation     object 
 13  Residence_Index       object 
 14  Foriegner_Index       object 
 15  Spouse_Index          object 
 16  Channel_Used          object 
 17  Deceased_Index        object 
 18  Primary_Address       float64
 19  Customer_Address      float64
 20  Province              object 
 21  Activity_Index        float64
 22  Gross_Income          float64
 23  Segmentation          object 
 24  Saving_Account        int64  
 25  Guarantees            int64  
 26  Current_Accounts      int64  
 27  Derivative_Account    int64  
 28  Payroll_Account       int64  
 29  Junior_Account        int64  
 30  More_Private_Account  int64  
 31  Private_Account       int64  
 32  Private_Plus_Account  int64  
 33  Short_Term_Deposits   int64  
 34  Medium_Term_Deposits  int64  
 35  Long_Term_Deposits    int64  
 36  E_Account             int64  
 37  Funds                 int64  
 38  Mortgage              int64  
 39  Pensions              int64  
 40  Loans                 int64  
 41  Taxes                 int64  
 42  Credit_Card           int64  
 43  Securities            int64  
 44  Home_Account          int64  
 45  Payroll               float64
 46  Pensions2             float64
 47  Direct_Debit          int64  
dtypes: float64(8), int64(23), object(17)
memory usage: 4.9+ GB
Changing datatypes
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw['Customer_Join_Date'] = pd.to_datetime(df_raw['Customer_Join_Date'])
df_raw['Customer_Leave_Date'] = pd.to_datetime(df_raw['Customer_Leave_Date'])

for column in ["Employee_Index", "Country", "Gender"]:
    df[column] = df[column].astype('category')
creating master data
Keeping just one data for each customer
df = df.drop_duplicates(subset='Customer_Code', keep="last")
Visual Insights
Types of customers


- There are greater number of Inactive Customers than Active Customer.
Number of Customers by Age


- XYZ Credit Unions' the greatest number of customers are in the Adult Age Group.
Adult age group and Number of Accounts


- Customers in the age of 40-50 are more likely to possess more than 10 different banking product at XYZ Credit Union.
Total Number of Different Types of Accounts


- The highest number of accounts sold are Current Accounts, Direct Debit, and Private Account; while the lowest sold accounts are Medium Term Deposits, Short Term Deposits, Derivative accounts, Savings Account and Guarantees.
Number of Customers by Gender


- There are more female customers than male customers in XYZ Credit Union.
Segmentation


- There are approximately 130,000 individuals have accounts with XYZ Credit Union. Nearly 20,000 VIP members are associated with the Union.
Top 10 channels


- Over a million customer have joined XYZ Credit Union through top 10 channels out of total 147 channels.
Corelation chart


- The above correlation chart shows that Payroll is highly related to Pensions2. And Payroll Account is highly related to Pensions2, Payroll, Debit and Credit Card.
Basic Insights
There are more number of Inactive Customers than Active Customer.
Some accounts are sold together such as Payroll is highly related to Pensions2 and Payroll Account is correlated with Pensions2, Payroll, Debit and Credit Card.
XYZ Credit Unions's most number of customers are in the Adult Age Group.
Customers in the age of 40-50 are more likely to possess more than 10 different banking product at XYZ Credit Union.
The highest number of accounts sold are Current Accounts, Direct Debit, and Private Account; while the lowest sold accounts are Medium Term Deposits, Short Term Deposits, Derivative accounts, Savings Account and Guarantees.
There are more female customers than male customers in XYZ Credit Union.
There are approximately 130,000 individuals have accounts with XYZ Credit Union. Nearly 20,000 VIP members are associated with the Union.
Customers with below average income are more than the customers with above average income.
The number of individual customers are more than the total number of college graduates and VIPs.
Over a million customer have joined XYZ Credit Union through top 10 channels out of total 147 channels.
Recommendations
Introducing loyalty programs such as health insurance or rewards for engaging with the account may increase the use of accounts that have been inactive for a while (dormant accounts).
Current account is the most selling banking product in XYZ Credit Union. For the customers having current accunt, scheme of gaining higher interest such as 4.5%, for keeping certain amount in savings account will increase sale of savings account.
Providing certain benefits to provincial and federal government for projects such as construction, may increase sale of Guarantees.
Engaging more with adults through social media coverage or advertising will help customers well understand product/services provided by XYZ Credit Union. Direct mail, email, statement inserts, banner ads on website, messages on ATMs, outbound calling campaigns, etc. can be applied as part of customer engagement.
Engaging more with the top 10 channels used by customers to join the Union will increases chances of getting more customers.
About
Configured & implemented an entire business understanding of banking products of a credit union to increase the cross-selling of products such as current accounts, savings account, credit cards, etc. Analyse the sales data through EDA process and prepared a Tableau Dashboard to dynamically represent corporate the insights and recommended solutions.

Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Jupyter Notebook
100.0%
Footer
