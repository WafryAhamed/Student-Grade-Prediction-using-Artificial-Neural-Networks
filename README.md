<h1 align="center">🎓 Student Grade Prediction using Artificial Neural Networks</h1>
<p align="center">
  Smart • Data-Driven • Optimized • Powered by Deep Learning
</p>



<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Bayesian-Optimization-purple" />
  <img src="https://img.shields.io/badge/Notebook-Google%20Colab-yellow?logo=googlecolab" />
</p>



## 📘 Project Overview

This project builds and optimizes an **Artificial Neural Network (ANN)** to predict student grades  
**(`A`, `B`, `C`, `F`)** using academic performance and engagement-based features.

The system integrates **Bayesian Optimization** for hyperparameter tuning, applies robust preprocessing (label encoding & normalization) and evaluates performance across **training, validation and test datasets**.



## ⚙️ Features Used

The ANN model is trained using the following engineered features:

1. **QuizAverage**  
   → Mean score of Quiz 1 and Quiz 2  

2. **StudyEfficiency**  
   → Ratio of study hours to academic credits  

3. **ParticipationScore**  
   → Attendance score plus engagement × 20  

4. **AssignmentPenaltyScore**  
   → Number of missed deadlines × assignment impact  
<img width="1920" height="1080" alt="model development  Tech Snatchers - Mini Project Presentation" src="https://github.com/user-attachments/assets/2afdac29-c56e-40f0-966f-18ed6c35cc7f" />
<img width="1920" height="1080" alt="model development  Tech Snatchers - Mini Project Presentation (1)" src="https://github.com/user-attachments/assets/c76634dc-089d-411d-a67a-564dffd8e273" />


<img width="1920" height="1080" alt="model development  Tech Snatchers - Mini Project Presentation (2)" src="https://github.com/user-attachments/assets/b4bf2a1e-afae-4534-bb84-39b51cb8bcd4" />

---

## 🚀 Workflow

1. **Load Datasets**  
   - Training set: 704 samples  
   - Validation set: 151 samples  
   - Test set: 31 samples  

2. **Preprocessing**  
   - Feature selection (`QuizAverage`, `StudyEfficiency`, etc.)  
   - Label encoding using `LabelEncoder`  
   - Feature normalization (mean and standard deviation saved)  

3. **Model Definition**  
   - Artificial Neural Network  
   - ReLU activation  
   - Batch Normalization and Dropout  
   - L2 regularization to reduce overfitting  

4. **Hyperparameter Tuning**  
   - Optimization performed using **BayesSearchCV**  
   - Tuned parameters:
     - Learning rate  
     - Dropout rate  
     - Number of neurons per layer  
     - Batch size  
     - Number of epochs  
   - Best configuration achieved approximately **0.85 cross-validation accuracy**  

5. **Final Model Training**  
   - Trained using optimized hyperparameters  
   - Callbacks applied:
     - `EarlyStopping`  
     - `ReduceLROnPlateau`  
   - Training history saved (accuracy and loss curves)  

6. **Evaluation**  
   - Performance evaluated on training and validation sets  
   - Test set evaluation using:
     - Accuracy  
     - Precision  
     - Recall  
     - F1-score  
   - Confusion Matrix generated for class-wise analysis  

7. **Model Saving**  
   - Final ANN model saved as `.h5`  
   - Normalization statistics and label encoder classes saved as `.npy` files  



## 📊 Evaluation Metrics

| Metric                  | Validation | Test |
|--------------------------|------------|------|
| Accuracy                | ~0.85      | ~0.84 |
| Macro Precision / Recall / F1 | ✅ | ✅ |
| Weighted Precision / Recall / F1 | ✅ | ✅ |

<img width="1920" height="1080" alt="final viva model development  Tech Snatchers - Mini Project Presentation" src="https://github.com/user-attachments/assets/54809ec7-4696-407b-8675-2ee8e58e8288" />



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
