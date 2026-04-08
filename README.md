# Classification with Support Vector Machines in Python
### Chapter 9 Applied Exercises — *An Introduction to Statistical Learning (ISLR2)*

[![Author](https://img.shields.io/badge/Author-Shad%20Ali%20Shah-blue)](https://github.com/shadalishah)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/shad-ali-shah-6439ab339/)
[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-Support%20Vector%20Machines-green)]()

---

## 🔍 What This Project Is About

This project demonstrates practical **binary classification** using Support Vector Machines (SVMs) — one of the most powerful and widely used machine learning algorithms in industry. SVMs are applied in fraud detection, medical diagnosis, image recognition, text classification, and many other real-world domains.

The exercises are drawn from **Chapter 9: Support Vector Machines** of *An Introduction to Statistical Learning with Applications in Python* (ISLR2) — a standard reference in data science and machine learning education worldwide.

> **In simple terms:** I trained algorithms to automatically sort observations into two groups — for example, high vs. low fuel efficiency, or one brand vs. another — and rigorously tested which model performs best on unseen data. This mirrors exactly what a data scientist does when building a classification system for a business problem.

---

## 💼 Skills Demonstrated

| Skill | What I Did |
|---|---|
| **Binary Classification** | Trained and evaluated classifiers on real and simulated datasets |
| **Support Vector Classifier (SVC)** | Applied linear kernel SVMs with regularization parameter tuning |
| **Kernel Methods** | Implemented polynomial and RBF (radial basis function) kernels for non-linear boundaries |
| **Logistic Regression (Linear & Non-linear)** | Compared standard and feature-engineered logistic regression against SVM |
| **Hyperparameter Tuning** | Used GridSearchCV to tune C, degree, and gamma across all kernel types |
| **Cross-Validation** | Applied k-fold CV to objectively identify the best model configuration |
| **Feature Scaling** | Pre-processed continuous features with StandardScaler before distance-based modelling |
| **Model Comparison** | Compared training error, CV error, and test error across multiple approaches |
| **Decision Boundary Visualization** | Produced classification boundary plots for each model type |
| **Python & Libraries** | `scikit-learn`, `statsmodels`, `pandas`, `numpy`, `matplotlib`, `ISLP` |

---

## 📂 Exercises Solved

---

### 🔵 Exercise 4 — *When does a non-linear decision boundary outperform a linear one?*

**Dataset:** Simulated two-class data (100 observations, 2 features, non-linear separation)

**Methods applied:** Linear SVC · Polynomial SVM (degree 4) · RBF SVM · Train/Test Split Evaluation

| Model | Training Error | Test Error |
|---|---|---|
| Linear SVC | 12% | 8% |
| Polynomial SVM (deg. 4) | **10%** | 8% |
| RBF SVM | 12% | **8%** |

**Key findings:**
- The **polynomial kernel** achieved the lowest training error (10%), confirming that non-linear kernels better capture the true separation structure in the training data.
- On the test set, **SVC (linear) and RBF SVM tied at 8% error** — showing that more complex kernels do not always outperform simpler ones on unseen data.
- Decision boundary plots visually confirmed that the polynomial and RBF kernels produce curved boundaries that more closely follow the non-linear class structure.

---

### 🔀 Exercise 5 — *Can logistic regression match an SVM on non-linear data?*

**Dataset:** Simulated data (n = 500, p = 2) with a quadratic decision boundary (Y = X₁² − X₂² > 0)

**Methods applied:** Linear Logistic Regression · Non-linear Logistic Regression (engineered features: X₁², X₁·X₂, log|X₂|) · Linear SVC · Polynomial SVM (degree 2)

**Key findings:**
- Standard **linear logistic regression** failed to recover the quadratic boundary — its decision surface was a straight line, misclassifying a large portion of observations.
- **Non-linear logistic regression** (with manually engineered quadratic features) dramatically improved classification, producing a curved boundary closely matching the true data structure.
- The **polynomial SVM** achieved the highest accuracy score, outperforming even the feature-engineered logistic regression — demonstrating SVMs' ability to capture complex boundaries without manual feature construction.
- Key insight: SVMs with appropriate kernels can automatically discover non-linear patterns that require explicit feature engineering in logistic regression.

---

### ⚖️ Exercise 6 — *Does a stricter or more lenient SVM generalize better?*

**Dataset:** Simulated two-class data (barely linearly separable, 100 observations)

**Methods applied:** Linear SVC · GridSearchCV over C = {0.1, 1, 10, 100, 1000} · 10-fold Cross-Validation · Independent Test Set Evaluation

**Key findings:**
- The **cross-validation error was minimized at C = 1** — a moderate regularization level that allows a small number of misclassifications on the training set.
- In contrast, large C values (100, 1000) achieved near-zero training errors but performed worse on the test set — a classic demonstration of **overfitting**.
- **C = 1 also minimized the test error**, confirming that the cross-validated choice generalizes well — a result that validates cross-validation as a reliable model selection tool.
- Key insight: for barely separable data, tolerating a few training misclassifications (lower C) consistently outperforms forcing a perfect training fit (higher C) in real-world settings.

---

### 🚗 Exercise 7 — *Can we predict whether a car gets high or low fuel efficiency?*

**Dataset:** Auto (392 vehicles; target: MPG above/below median → binary label)

**Methods applied:** Linear SVC · Polynomial SVM · RBF SVM · GridSearchCV with StandardScaler preprocessing

**Best parameters found:**

| Kernel | Best C | Best Other Param | CV Error |
|---|---|---|---|
| Linear | 1 | — | **8.66%** |
| Polynomial | 100 | degree = 5 | — |
| RBF | 1 | gamma = 0.5 | — |

**Key findings:**
- **Feature scaling** was applied before fitting (displacement, horsepower, weight, acceleration standardized) — essential for distance-based SVMs to prevent high-magnitude features from dominating.
- The **linear SVM with C = 1** achieved the lowest cross-validation error of 8.66%, making it the preferred model for this dataset.
- Polynomial and RBF kernels required broader hyperparameter searches but did not meaningfully improve over the linear baseline — suggesting the class boundary in this dataset is approximately linear in the feature space.
- Decision boundary plots for each kernel type were generated across feature pairs, visualizing how each model separates high- and low-MPG vehicles.

---

### 🍊 Exercise 8 — *Can we predict which orange juice brand a customer will buy?*

**Dataset:** OJ — Orange Juice (1,070 observations; target: Purchase = CH or MM)

**Methods applied:** Linear SVC · RBF SVM · Polynomial SVM (degree 2) · 5-fold CV with GridSearchCV (C ∈ {0.01, 0.1, 1, 10, 100})

**Full results summary:**

| Model | Initial C | Best C (CV) | Best Train Error | Best Test Error |
|---|---|---|---|---|
| Linear SVC | 0.01 | CV-tuned | Improved | ~18.15% |
| RBF SVM | 0.01 | CV-tuned | Improved | Comparable |
| Polynomial SVM (deg. 2) | 0.01 | CV-tuned | Improved | Comparable |

**Key findings:**
- At the initial C = 0.01, **435 support vectors** were required for the linear classifier — indicating a very soft margin with heavy regularization.
- After cross-validation tuning, all three kernels converged to **similar test errors**, with the **linear kernel narrowly achieving the best test error of ~18.15%**.
- This consistency across kernels suggests the OJ purchase decision boundary is well-approximated by a linear function of the available features.
- Key insight: investing in more complex kernels does not always pay off — for this dataset, the simpler linear model is more efficient and equally accurate.

---

## 📁 Repository Structure

```
📦 ISLR2-Chapter9-SupportVectorMachines/
│
├── 📓 Chapter_9.ipynb     # Full Python notebook with code + explanations
├── 🌐 chapter_9.html      # Web-viewable version of the notebook
├── 📄 chapter_9.qmd       # Source file (Quarto format)
└── 📋 README.md           # This file
```

---

## ▶️ How to Run This Project

**Step 1 — Install Python dependencies:**
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels islp jupyter
```

**Step 2 — Clone the repository:**
```bash
git clone https://github.com/shadalishah/ISLR2-Chapter9-SupportVectorMachines.git
cd ISLR2-Chapter9-SupportVectorMachines
```

**Step 3 — Open the notebook:**
```bash
jupyter notebook Chapter_9.ipynb
```

---

## 🙏 Acknowledgements

Special thanks to **[Karim Aboussel Ham](https://github.com/KarimABOUSSELHAM)** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM/ISLP-applied-solutions) provided helpful code examples and guidance during the completion of these exercises.

---

## 👤 About the Author

**Shad Ali Shah**
MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
Passionate about the intersection of **Economics, Data Science, and Machine Learning**

🔗 [LinkedIn](https://www.linkedin.com/in/shad-ali-shah-6439ab339/) &nbsp;|&nbsp; 🐙 [GitHub](https://github.com/shadalishah)

---

## 📚 Reference

> James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
> *An Introduction to Statistical Learning with Applications in Python*. Springer.
> [https://www.statlearning.com](https://www.statlearning.com)

---

*This project is for academic and portfolio purposes.*
