# DevSecMLOps Lab

This repository contains the implementation of DevSecMLOps laboratory experiments completed as part of the course.

## Experiments Completed

### Experiment 2 – Version Control for ML Projects

**Objective:**  
To apply Git branching, committing, and merging workflow to a machine learning project.

**Technologies Used:**
- Python
- Scikit-learn
- Git
- GitHub

**ML Model:**
- Decision Tree Classifier
- Dataset: Iris Dataset

**Results:**
- Accuracy: 0.97
- Precision: 0.97
- Recall: 0.97

**File:**
`Experiment2/train.py`

---

### Experiment 3 – End-to-End Automated Data Pipeline

**Objective:**  
To implement an automated machine learning pipeline consisting of:

Extract → Validate → Transform → Train → Evaluate → Persist

**Technologies Used:**
- Python
- Pandas
- Scikit-learn
- Joblib

**ML Model:**
- Random Forest Classifier
- Dataset: Iris Dataset

**Pipeline Stages:**
1. Extract the dataset
2. Validate the data
3. Standardize features
4. Train the model
5. Evaluate the model
6. Save the model and scaler

**Results:**
- Dataset: 150 rows × 5 columns
- Features standardized: 4
- Test Accuracy: 0.933

**Generated Artifacts:**
- `model.pkl`
- `scaler.pkl`

**File:**
`Experiment3/exp3_pipeline.py`

---

## Project Structure

```text
DevSecMLOps-Lab/
│
├── Experiment2/
│   └── train.py
│
├── Experiment3/
│   ├── exp3_pipeline.py
│   └── artifacts/
│       ├── model.pkl
│       └── scaler.pkl
│
└── README.md
