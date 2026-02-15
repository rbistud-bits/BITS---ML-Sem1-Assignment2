# ML Classification Benchmark Web App

**BITS Pilani – Machine Learning Assignment 2**

---

# 1. Problem Statement

The objective of this assignment is to build an end-to-end Machine Learning workflow that:

• Implements multiple classification algorithms
• Evaluates them using standard metrics
• Deploys an interactive Streamlit web application

The application allows users to upload any classification dataset and compare model performance automatically.

This demonstrates the complete ML lifecycle:
**Data → Modeling → Evaluation → Deployment**

---

# 2. Dataset Description 

### Dataset Used

**Breast Cancer Wisconsin Dataset (UCI / Scikit-Learn)**

This dataset is widely used for binary classification tasks to predict whether a tumor is **Malignant (M)** or **Benign (B)**.

### Dataset Summary

| Property                | Value                  |
| ----------------------- | ---------------------- |
| Instances               | 569                    |
| Features after encoding | 31                     |
| Target Variable         | diagnosis              |
| Classes                 | 2 (Malignant / Benign) |

The dataset satisfies assignment constraints:

* Minimum rows ≥ 500 ✔
* Minimum features ≥ 12 ✔

---

# 3. Models Implemented 

The following **6 classification models** were implemented and evaluated on the SAME dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

# 4. Evaluation Metrics Used

Each model is evaluated using:

| Metric            | Description                        |
| ----------------- | ---------------------------------- |
| Accuracy          | Overall correctness                |
| AUC Score         | Area Under ROC Curve               |
| Precision (macro) | Correct positive predictions       |
| Recall (macro)    | Ability to detect positives        |
| F1 Score (macro)  | Precision-Recall balance           |
| MCC Score         | Balanced metric for classification |

---

# 5. Model Comparison Table 

## Breast Cancer Dataset Results

| ML Model            | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | **0.9649** | 0.9960     | 0.9672     | 0.9573     | 0.9619     | 0.9245     |
| Decision Tree       | 0.9386     | 0.9282     | 0.9360     | 0.9315     | 0.9337     | 0.8676     |
| KNN                 | 0.9561     | 0.9828     | 0.9605     | 0.9454     | 0.9521     | 0.9058     |
| Naive Bayes         | 0.6228     | 0.9246     | 0.3142     | 0.4931     | 0.3838     | -0.0718    |
| Random Forest       | **0.9737** | 0.9934     | 0.9800     | 0.9643     | 0.9713     | 0.9442     |
| XGBoost             | **0.9737** | **0.9954** | **0.9800** | **0.9643** | **0.9713** | **0.9442** |

---

# 6. Observations

| Model               | Observation                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| Logistic Regression | Strong baseline model with excellent AUC and balanced performance.     |
| Decision Tree       | Performs well but slightly prone to overfitting compared to ensembles. |
| KNN                 | Good performance but sensitive to feature scaling and dataset size.    |
| Naive Bayes         | Assumption of feature independence not suitable → weakest performer.   |
| Random Forest       | Excellent accuracy and MCC due to ensemble learning.                   |
| XGBoost             | **Best overall model** with highest AUC and balanced metrics.          |

### Final Conclusion

**Ensemble models (Random Forest & XGBoost) clearly outperform other models.**

---

# 7. Streamlit Web App Features

The deployed app allows users to:

✔ Upload any CSV classification dataset
✔ Automatically detect target variable
✔ Train all 6 ML models
✔ Compare evaluation metrics
✔ View Confusion Matrix & Classification Report
✔ Visualize ROC Curve
✔ Download results

This satisfies Streamlit requirements:

* Dataset upload ✔
* Model dropdown ✔
* Metrics display ✔
* Confusion matrix ✔

---

# 8. How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 9. Deployment

The application is deployed on **Streamlit Community Cloud**.

Live App Link: *(Add after deployment)*

---

# 10. Repository Structure

```
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── __init__.py
    └── models.py
```

---
