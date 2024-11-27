# Model Card
Prediction of particle size and yield for continuous crystallisation of API using XGBoost model
## Model Description
*Input:**  
The model takes the following features as input:  
- `V_stage1`, `V_stage2`, `V_stage3`: Volumes at different stages of the crystallization process.  
- `T_stage1`, `T_stage2`, `T_stage3`: Temperatures at corresponding stages.  
- `feed_flowrate`: Flow rate of feed material into the crystallizer.  
- `seed_flowrate`: Flow rate of seed material used in the crystallization process.
  
**Output:**  
The model predicts two key outputs:  
1. **Yield**: The efficiency of the crystallization process, expressed as a percentage of material crystallized.  
2. **Particle Size**: The size of the crystals formed, which is critical for product quality.  

**Model Architecture:**  
The model is built using **XGBoost**, an ensemble learning algorithm based on decision trees. The key parameters used include:
- Objective: `reg:squarederror` (for regression tasks).
- Evaluation Metric: Root Mean Squared Error (RMSE).
- Hyperparameters: 
  - Learning Rate: 0.1  
  - Maximum Depth: 6  
  - Early Stopping: 10 rounds  

The dataset is split into training (60%), validation (20%), and testing (20%) sets, with features normalized using `MinMaxScaler`.

## Performance
he model’s performance metrics were evaluated using RMSE (lower is better) and R² (higher is better, with 1.0 indicating perfect fit). Results are summarized below:

### Yield
- **Train RMSE:** 0.053  
- **Train R²:** 0.999  
- **Validation RMSE:** 0.085  
- **Validation R²:** 0.996  
- **Test RMSE:** 0.115  
- **Test R²:** 0.992  

### Particle Size
- **Train RMSE:** 5.266  
- **Train R²:** 0.989  
- **Validation RMSE:** 7.533  
- **Validation R²:** 0.980  
- **Test RMSE:** 7.731  
- **Test R²:** 0.979  

### Learning Curve Example (Yield)
The RMSE values during training and validation are shown below:  

![Learning Curve](path/to/learning_curve.png)
## Limitations
*Data Specificity:**  
   The model is trained on a specific dataset of crystallization experiments and may not generalize well to significantly different setups or conditions.
**Feature Dependence:**  
   The model assumes the provided features are accurate and well-measured. Noisy or incomplete data could impact predictions.
**Output Dependency:**  
   Predictions for `yield` and `particle_size` are made independently. Interaction between these targets is not explicitly modeled.
## Trade-offs
**Overfitting vs. Generalization:**  
   While the model achieves high performance on the training and validation sets, slight degradation in R² on the test set for particle size indicates potential overfitting.
**Complexity vs. Interpretability:**  
   Although XGBoost is highly effective, its decision-tree-based ensemble makes it less interpretable compared to simpler models like linear regression.
**Performance Variability:**  
   The model may exhibit reduced performance for unseen conditions, particularly for `particle_size`, which is more complex to predict.