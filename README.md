# Loan Default Prediction using Logistic Regression

This project applies **Logistic Regression** to predict loan default on an imbalanced dataset.  
Since class imbalance is a common problem in financial datasets, we used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training data.  

The model was then trained, evaluated, and compared using classification metrics such as Accuracy, Precision, Recall, and F1-score.

---

## 📊 Project Workflow

1. **Data Preprocessing**
   - Handled missing values
   - Feature scaling using `MinMaxScaler`
   - Train-test split with `random_state=42` for reproducibility

2. **Handling Imbalance**
   - Applied **SMOTE** on the training set
   - Ensured test set remained untouched for fair evaluation

3. **Model Training**
   - Logistic Regression (`solver='lbfgs'`, `penalty='l2'`, `max_iter=200`, `random_state=42`)
   - Compared performance with/without SMOTE

4. **Evaluation**
   - Confusion Matrix
   - Accuracy
   - Precision, Recall, F1-score
   - Classification Report
   - Visualization using Seaborn

---

## 📈 Results

- **Accuracy:** ~81%
- **Class 0 (Non-default) Recall:** ~57%
- **Class 1 (Default) Recall:** ~91%

While the model performs well for detecting defaults, recall for non-default cases is relatively low.  
This highlights the limitation of Logistic Regression for complex, imbalanced datasets and motivates trying **Random Forest, XGBoost, or CatBoost** in future improvements.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 🚀 Future Improvements
- Try **tree-based models** (Random Forest, XGBoost, CatBoost)  
- Apply **threshold tuning** for better recall on minority class  
- Perform **hyperparameter tuning** with GridSearchCV/RandomizedSearchCV  
- Add **cross-validation** for more stable evaluation  

---

## 📂 How to Run
```bash
git clone https://github.com/<your-username>/loan_default_prediction_logistic_regression.git
cd loan_default_prediction_logistic_regression
jupyter notebook Big_project_28_aug.ipynb
📜 License
This project is licensed under the MIT License.
