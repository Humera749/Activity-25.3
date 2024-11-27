# Prediction of process yield and particle size for continuous crystallisation of API using Ensemble methods
## Project overview
The aim of this project is to predict the performance of a continuous crystallisation process by investigating the relationship between input and output variables  . Key objectives from this project were to predict yield and particles size (output functions) (required critical attributes) by investigating the effect of process and operational variables (input functions). These process and operational variables are volume of each stage of crystallisation, temperature of stages of crystallisation, feed and seed flowrate. it is important to understand which variable has most significant effect on the output functions. Interpretability of the model is crucial.
## Data
This dataset was developed as part of a project involving the design of a continuous crystallization process for an active pharmaceutical ingredient (API). Initially, a mechanistic model was established for the batch process, and estimated parameters from the batch model were used to design the continuous process. The dataset was simulated by applying various operational and physical constraints to the system to explore how these parameters and constraints influence the final outcomes.
The study focused on a three-stage crystallization process, and the dataset includes key parameters such as:
Volume of each stage of crystallization: The stage volume determines the residence time, which is critical for achieving the desired yield.
Temperature of each stage: Temperature influences the driving force for crystallization, affecting both crystal growth and yield, making it a crucial parameter for particle size and yield optimization.
Feed flowrate: Feed flowrate impacts residence time and is a key determinant of the process yield.
Seed flowrate: Seed flowrate plays a pivotal role in crystal growth and significantly affects particle size.
This structured approach aimed to analyze the relationships between these parameters and the resulting process outcomes, providing insights for optimizing continuous crystallization performance.
## Model
As mentioned earlier interpretability is the key focus and crystallization processes are often governed by nonlinear kinetics and thermodynamics, Ensemble methods are the best choice as a starting point.  I started with random forest model and then use XGBoost and lightLGM to improve predictions.
### Model 1: Random forest
This code implements machine learning workflow to predict particle size and yield in a continuous crystallization process using a Random Forest Regressor. The dataset, read from an Excel file, contains process parameters (stage volumes, stage temperatures, feed, and seed flow rates) as features, and particle size and yield as targets. The features are normalized using MinMaxScaler, and the dataset is split into training, validation, and test sets for both targets. Two Random Forest models are trained separately for particle size and yield.
The evaluation includes computing Root Mean Square Error (RMSE) and R² scores on training, validation, and test sets, with results plotted as learning curves, prediction vs. actual comparisons, residuals, and feature importance visualizations. Key plots and metrics provide insights into the model's performance and the impact of features. This structured workflow emphasizes interpretability and ensures robust validation through systematic analysis of errors and featu and contribution.
### Model 2: XGBoost
This code implements a machine learning workflow using XGBoost to predict yield and particle size data preprocessing and splits are done in similar manner as mentioned for model 1.
For model training, the XGBoost Regressor is configured with the following hyperparameters:
•	Objective: reg:squarederror (for regression tasks)
•	Eval Metric: rmse (Root Mean Squared Error)
•	Learning Rate: 0.1
•	Max Depth: 6
•	Seed: 42
The model is trained using early stopping with a maximum of 200 boosting rounds and a patience of 10 rounds to prevent overfitting. Performance is evaluated using RMSE and R² metrics on training, validation, and test sets.
The workflow generates multiple visualizations, including:
1.	Learning Curves: RMSE evolution over boosting rounds for training and validation sets.
2.	Prediction vs Actual Plots: Comparing predicted values with actual values to assess accuracy.
3.	Residual Plots: Highlighting prediction errors for model diagnostics.
4.	Feature Importance: Ranking features based on their impact on model predictions.
This comprehensive approach ensures robust validation and interpretability of the predictive models for both yield 
### Model 3: LightLGM
Similar to XGBoost, LightGBM uses gradient boosting to build trees sequentially. Instead of growing trees level by level (as XGBoost does), LightGBM grows trees leaf-wise. This means it expands the leaf with the highest loss, leading to deeper and more optimal trees. Uses histogram-based splitting for faster training. Handles categorical features directly without one-hot encoding it gives better performance on high dimension data and faster training as compared to XGBoosta## Hyperparameter optimisation
I applied manual tuning of hyper parameters in random forest model for hyperparameters number of trees, depth of trees and min_sample_split and min_sample_leaf. Lateron i used Optuna and baysian optimisation to tune hyperparameters for XGBoost model (my preferred model based on outcomes).
### Optuna
Separate objective functions for Particle Size and Yield use Optuna to optimize key XGBoost parameters, including learning_rate, max_depth, min_child_weight, subsample, and colsample_bytree, over 50 trials to minimize RMSE on validation sets. The tuned models are trained with the best hyperparameters and evaluated on test sets using RMSE. The process includes visualization of optimization history and final parameter selection. This efficient approach highlights automated optimization and robust evaluation of XGBoost's predictive performance.
### Baysian optimisation
Bayesian Optimization optimizes key XGBoost parameters, such as learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, and n_estimators, by maximizing the negative RMSE on the validation set. Separate objective functions are defined for Particle Size and Yield, and the optimizations run over 50 iterations with 5 initialization points. The best parameters for each target are extracted, and final models are trained and evaluated on test sets, achieving performance metrics like RMSE. This code showcases an efficient, probabilistic hyperparameter optimization method for robust regression tasks.
## Results
### Random forest model 
provided good predictions for particle size and yield.below are the results for the model. to improve generalisation of model for particle size XGBoost model was tested. AS RMSE for train, test and validation wasnot converged well for Random Forest model. Some manual tuning of hyper parameters was also performed increasing number of trees and max depth of trees.
Particle Size - Performance Metrics:
Train RMSE: 2.778, Train R²: 0.997
Validation RMSE: 7.964, Validation R²: 0.977
Test RMSE: 8.522, Test R²: 0.
Yield - Performance Metrics:
Train RMSE: 0.020, Train R²: 1.000
Validation RMSE: 0.055, Validation R²: 0.998
Test RMSE: 0.043, Test R²: 0.
To check the robustness of the model cross validation was performed using K-Fold approach. Model performance was very close for all trials showing a robust model performance.
Particle Size - Cross-Validation Results:
Fold 1: RMSE = 8.640, R² = 0.973
Fold 2: RMSE = 7.197, R² = 0.981
Fold 3: RMSE = 7.142, R² = 0.980
Fold 4: RMSE = 7.974, R² = 0.974
Fold 5: RMSE = 7.440, R² = .979

Aggregate Metrics: Average RMSE = 7.679, Average R² = 0.977
### XGBoosModel gave similar performance to random forest model however convergence of RMSE for particle size was better with this model.
Yield - Performance Metrics:
Train RMSE: 0.053, Train R²: 0.999
Validation RMSE: 0.085, Validation R²: 0.996
Test RMSE: 0.115, Test R²: 0
Particle Size - Performance Metrics:
Train RMSE: 5.266, Train R²: 0.989
Validation RMSE: 7.533, Validation R²: 0.980
Test RMSE: 7.731, Test R²: 0
### LIGHTLGM and hyperparameter tuning
no further improvement was observed using LightLGM or by hyperparameter tuning using Optuna and baysian optimisation. model already showing good performance with the current hyper parameter settings..979.992t
This 999974nd particle size.
