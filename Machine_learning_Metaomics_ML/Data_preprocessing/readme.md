# Data Preprocessing in Machine Learning

Data preprocessing is a crucial step in the machine learning (ML) pipeline.  
It involves transforming raw data into a clean and usable format, ensuring better model performance and accuracy.  

---
<img width="987" height="407" alt="image" src="https://github.com/user-attachments/assets/a120afc9-229c-4c20-98f8-bfea9e383e56" />

##  General Steps in Data Preprocessing

---

## 1. Data Cleaning 
- Handle **missing values** (remove, impute with mean/median/mode).
- Remove **duplicates**.
- Detect and handle **outliers**.
- Ensure data consistency ( formats, naming).

---

## 2. Dimensionality Reduction 
- Reduces the number of input features while retaining important information.
- Techniques:
  - **PCA (Principal Component Analysis)**
  - **t-SNE (t-distributed Stochastic Neighbor Embedding)**
  - **Autoencoders**
- Benefits:
  - Speeds up training
  - Reduces overfitting
  - Improves visualization of data

---

## 3. Feature Engineering 
- Creating new features from existing ones to improve model performance.
- Examples:
  - Extracting **collective features**.
  - Combining attributes.
  - Domain-specific transformations.

---

## 4. Sampling Data 
- Selecting a subset of data to train the model efficiently.
- Techniques:
  - **Random Sampling**: Select random records.
  - **Stratified Sampling**: Preserve class proportions.
  - **Bootstrapping**: Sampling with replacement for ensemble models.

---

## 5. Data Transformation 
- Scaling and normalizing features:
  - **Normalization (Min-Max Scaling)** → Values between 0 and 1.
  - **Standardization (Z-score scaling)** → Mean = 0, Std Dev = 1.
- Encoding categorical variables:
  - **Label Encoding**
  - **One-Hot Encoding**
- Log transformations or Box-Cox to handle skewed distributions.

---

## 6. Handling Imbalanced Data 
- Problem: Some classes have many more samples than others.
- Solutions:
  - **Oversampling** minority class (e.g., SMOTE).
  - **Undersampling** majority class.
  - Using **class weights** in algorithms.
  - Ensemble methods (e.g., Balanced Random Forest).
