
# LendingClub Risk Prediction

Risk assessment plays a crucial role in financial services. By utilizing a real-world dataset from LendingClub, this project focuses on predictive analytics to forecast loan defaulters. The goal is to identify risky loan applicants, understand the factors driving default/charged off, and enable informed decision-making. The project involves exploratory data analysis, preprocessing, analysis of different models, data balancing techniques, and hyperparameter tuning to enhance performance and optimize predictions.

## Acknowledgements

I would like to extend my heartfelt gratitude to the ICT Academy of Kerala for providing the platform and resources for this final project. The support and opportunities offered by the academy have been invaluable in shaping our skills and fostering our growth as a team. Additionally, I would like to express my appreciation to Mr.Sandeep, my mentor from the ICT Academy, for their guidance and encouragement throughout the project. 
## Python Packages Used

Data Manipulation: pandas, numpy, and other relevant packages for efficient importing and handling of data.
Data Visualization: Utilized packages like seaborn, matplotlib, and others for creating informative graphs and visualizations during the analysis process and for understanding the ML models.
Machine Learning:Implemented packages such as scikit-learn and TensorFlow to develop and train machine learning models. 
## Dataset

For this project, a real-world dataset from Kaggle was used. The dataset can be accessed through the following link: https://www.kaggle.com/datasets/epsilon22/lending-club-loan-two
 This dataset provides authentic and representative data that was crucial for conducting the analysis and developing accurate models.


## Exploratory Data Analysis

The EDA encompassed a broad summary and key statistics of the dataset, providing initial insights. Visualizations including pie charts, bar charts, distribution plots, and box plots were employed to analyze the data. 
## Data Preprocessing

The initial dataset exhibited irregularities that required comprehensive data preprocessing. Measures were taken to address missing values, outliers, and achieve normalization for optimal modeling.

Missing values were handled through removal techniques to ensure data completeness. Outliers, which could introduce bias or skewness, were identified and treated appropriately to minimize their influence on the modeling process.

Label encoding was applied to categorical variables to convert them into numerical representations, enabling the models to process the data effectively.

Normalization techniques were applied to standardize the feature scales, promoting fair comparison and preventing any single feature from dominating the model due to its magnitude.

By diligently performing these essential data preprocessing tasks, the dataset was refined to uphold its quality and integrity, laying a solid foundation for accurate and robust modeling.
## Results and evaluation

Rather than relying solely on high accuracy, which can be misleading for imbalanced data, the evaluation of models focused on identifying those with the highest F1-Score and ROC-AUC values. These metrics were deliberately chosen as alternative measures to effectively handle the challenges posed by imbalanced datasets.

Here are the F1-Score and ROC values for different machine learning models before and after hyper parameter tuning:

- Before Tuning:

Logistic Regression with near miss: F1-Score = 0.38, ROC = 0.598
Logistic Regression with SMOTE: F1-Score = 0.43, ROC = 0.654
Logistic Regression with cost-sensitive learning: F1-Score = 0.43, ROC = 0.654
Decision Tree with near miss: F1-Score = 0.33, ROC = 0.538
Decision Tree with SMOTE: F1-Score = 0.29, ROC = 0.550
Decision Tree with bagging: F1-Score = 0.20, ROC = 0.536
Decision Tree with cost-sensitive learning: F1-Score = 0.28, ROC = 0.551
Random Forest with near miss: F1-Score = 0.35, ROC = 0.557
Random Forest with SMOTE: F1-Score = 0.29, ROC = 0.654
Neural Network: F1-Score = 0.34, ROC = 0.718
Neural Network with cost-sensitive learning: F1-Score = 0.34, ROC = 0.718

- After Tuning:

RandomsearchCV applied to Random Forest with near miss: F1-Score = 0.37, ROC = 0.588
RandomsearchCV applied to Random Forest with SMOTE: F1-Score = 0.41, ROC = 0.636
Kerastuner applied to Neural Network with cost-sensitive learning: F1-Score = 0.34, ROC = 0.657


Based on the results, the Random Forest with SMOTE model achieved the highest F1-Score of 0.41 after tuning, leading to its selection for building the application, which was then hosted using Flask.
## Future Work

- Feature Engineering: Explore additional feature engineering techniques to enhance the predictive power of the models. Consider generating new features or deriving more informative variables from the existing ones.

- Advanced Modeling Techniques: Investigate other machine learning algorithms such as support vector machines (SVM). Assess their performance and potential impact on the prediction accuracy.

- Hyperparameter Optimization: Conduct more extensive hyperparameter tuning to find optimal configurations for the chosen models. Utilize techniques like grid search, or Bayesian optimization to systematically search the hyperparameter space.

- Imbalanced Data Handling: Investigate additional techniques specifically designed for handling imbalanced data, such as adaptive synthetic sampling (ADASYN), or SMOTE variants. Assess their impact on improving model performance and mitigating the effects of class imbalance.

- External Data Sources: Consider integrating external data sources, such as economic indicators or industry-specific data, to enrich the existing dataset and potentially uncover additional patterns and relationships.

- Collaborative Research: Collaborate with domain experts or researchers in the field to gain further insights into the problem domain and explore novel approaches to address specific challenges in loan default prediction.
## References

- Cost-Sensitive Learning and the Class Imbalance Problem  -   Charles X. Ling, Victor S. Sheng, The University of Western Ontario, Canada
- Predicting Default Risk on Peer-to-Peer Lending Imbalanced Datasets   -   Yen-Ru Chen1, Jenq-Shiou Leu 1, Sheng-An Huang1, Jui-Tang Wang 1, And Jun-Ichi Takada2, Tokyo Institute of Technology, Japan
