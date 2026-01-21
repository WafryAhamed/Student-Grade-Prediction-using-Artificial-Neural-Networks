# 🎓 Student Grade Prediction using Artificial Neural Networks

This project develops and optimizes an **Artificial Neural Network (ANN)** to predict student grades (`A`, `B`, `C`, `F`) using academic and engagement features.  
It integrates **Bayesian Optimization** for hyperparameter tuning, applies preprocessing (label encoding + normalization), and evaluates performance on training, validation, and test datasets.

In addition, the project implements a **complete end-to-end pipeline** from raw data preprocessing to real-time student grade prediction.

---

## 🧠 System Pipeline Overview (Added)

Raw student data → Data cleaning → Feature engineering → Label encoding → Normalization →  
Train / Validation / Test split → ANN training → Hyperparameter tuning → Model evaluation →  
Model saving → New student input → Preprocessing → Grade prediction

This ensures consistency between training and real-world usage.



---

## 📊 Evaluation Metrics

| Metric                  | Validation | Test |
|--------------------------|------------|------|
| Accuracy                | ~0.85      | ~0.84 |
| Macro Precision / Recall / F1 | ✅ | ✅ |
| Weighted Precision / Recall / F1 | ✅ | ✅ |



## 📈 Results
- **Best ANN model** tuned with Bayesian Optimization achieved **~85% accuracy** on test data.  
- Feature engineering (StudyEfficiency, AssignmentPenaltyScore) boosted performance.  
- Normalization was critical for stable training.  



## 🛠️ Technologies
- **Python** (NumPy, Pandas, Matplotlib)  
- **Scikit-learn, Scikit-optimize, SciKeras**  
- **TensorFlow / Keras**  
- **Google Colab (GPU runtime)**  



## 👨‍💻 Authors
**Tech Snatchers (FAS)** – Rajarata University of Sri Lanka  
