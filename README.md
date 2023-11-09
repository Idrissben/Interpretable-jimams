# Algorithmic Fairness and Interpretability - JIMAMS

This repository hosts the source code for developing a predictive model to determine an individual's parole grant status, utilizing a comprehensive dataset encompassing various features such as sex, criminal history, ethnicity, and age. The project also delves into the rigorous evaluation of the model's fairness and interpretability, applying principles of algorithmic fairness and transparency.

## Team JIMAMS
This endeavor was carried out by Team JIMAMS as part of the 'Algorithmic Fairness and Interpretability' course. The dedicated team members contributing to this project include:

- Idriss Bennis
- Ajouad Akjouj
- Mathieu Péharpré
- Joseph Moussa
- Samuel Berrebi
- Marie-Sophie Richard

## Project Steps

- **Step 1**: Dataset Description and Exploratory Data Analysis
  In this initial phase, the dataset is meticulously described, and an exploratory data analysis is conducted to gain a comprehensive understanding of the data's characteristics.

- **Step 2**: XGBoost Model Training
  The XGBoost machine learning model is trained on the dataset to predict parole outcomes based on the available features.

- **Step 3**: Comparison with White Box Models
  A comparative analysis is performed to contrast the XGBoost model's performance with that of interpretable white box models.

- **Step 4**: Surrogate Method for XGBoost Interpretation
  To enhance interpretability, a surrogate method is employed to provide insights into the decision-making process of the XGBoost model.

- **Step 5**: PDP and ALE for XGBoost Interpretation
  Partial Dependence Plots (PDP) and Accumulated Local Effects (ALE) are used to unveil the relationships between individual features and model predictions for improved interpretation.

- **Step 6**: Post-Hoc Local Methods for Local Interpretability
  Post-hoc local interpretability methods are applied to gain a more granular understanding of model predictions for specific instances.

- **Step 7**: Performance Interpretability Using Permutation Importance and XPER
  The model's performance is assessed using techniques such as permutation importance and the XPER metric to ascertain feature importance and model behavior.

- **Step 8**: Model Fairness with Respect to Protected Attributes and Mitigation
  A comprehensive examination of model fairness is conducted, with a specific focus on the impact of protected attributes, and potential mitigation strategies are explored.

- **Step 9**: FPDP Implementation Using a Fairness Measure
  A Fairness-Preserving Decision Process (FPDP) is implemented, guided by a fairness measure, to ensure that the model's predictions exhibit fairness with regard to sensitive attributes.