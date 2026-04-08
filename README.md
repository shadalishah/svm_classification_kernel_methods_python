# ⚡ Support Vector Machines — Binary Classification with Linear, Polynomial & RBF Kernels

> **Skills Demonstrated:** Support Vector Machines · SVC · Kernel Methods · RBF · Polynomial Kernel · Hyperparameter Tuning · GridSearchCV · Cross-Validation · Feature Scaling · Decision Boundary Visualization · Logistic Regression · Python · Scikit-learn

---

## 🎯 Project Overview

This project implements **Support Vector Machines (SVMs)** for binary classification across five progressively complex real and simulated datasets — from linearly separable data to non-linear quadratic boundaries to real-world consumer purchase prediction.

> *"SVMs are deployed in fraud detection, medical diagnosis, image recognition, text classification, and credit scoring. This project demonstrates the full SVM workflow — from kernel selection to hyperparameter tuning to test evaluation."*

Five exercises are covered:

| Exercise | Dataset | Business Problem |
|----------|---------|-----------------|
| Ex. 4 | Simulated (non-linear) | When does a kernel SVM outperform a linear one? |
| Ex. 5 | Simulated (quadratic) | Can logistic regression match SVM on non-linear data? |
| Ex. 6 | Simulated (barely separable) | Strict vs lenient margin — which generalizes better? |
| Ex. 7 | Auto (Real) | Predict high vs low fuel efficiency for 392 vehicles |
| Ex. 8 | OJ (Real) | Predict orange juice brand purchase (CH vs MM) |

---

## 📁 Datasets Used

| Dataset | Source | Size | Target Variable |
|---------|--------|------|----------------|
| **Simulated (Ex. 4)** | `numpy.random` | 100 obs, 2 features | Binary class (non-linear boundary) |
| **Simulated (Ex. 5)** | `numpy.random.default_rng(5)` | 500 obs, 2 features | Y = X₁² − X₂² > 0 (quadratic) |
| **Simulated (Ex. 6)** | `numpy.random` | 100 obs, 2 features | Binary class (barely separable) |
| **Auto** | Carnegie Mellon StatLib (Real) | 392 vehicles, 7 features | mpg01 — High/Low Fuel Efficiency |
| **OJ** | Sales Scan Data (Real) | 1,070 obs, 18 features | Purchase — CH or MM brand |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Purpose |
|-----------|---------|---------|
| Support Vector Classifier (Linear SVC) | `sklearn.svm.SVC(kernel='linear')` | Linear decision boundary |
| Polynomial Kernel SVM | `SVC(kernel='poly', degree=d)` | Curved polynomial boundaries |
| RBF Kernel SVM | `SVC(kernel='rbf', gamma=g)` | Radial basis non-linear boundaries |
| Logistic Regression (Linear) | `sklearn.linear_model.LogisticRegression` | Linear baseline comparison |
| Non-Linear Logistic Regression | Manual feature engineering (X₁², X₁·X₂, log\|X₂\|) | Quadratic boundary via feature engineering |
| GridSearchCV | `sklearn.model_selection.GridSearchCV` | Tuning C, degree, gamma |
| K-Fold Cross-Validation | `sklearn.model_selection.cross_val_score` | Objective model selection |
| Feature Scaling | `sklearn.preprocessing.StandardScaler` | Essential for distance-based SVMs |
| Decision Boundary Plots | `matplotlib` + `plot_svm()` | Visual model comparison |
| Train/Test Split | `train_test_split` | Honest out-of-sample evaluation |

**Libraries:** `numpy` · `pandas` · `scikit-learn` · `matplotlib` · `statsmodels` · `ISLP`

---

## 📊 Key Results

### Exercise 4 — Non-Linear Data: When Does Kernel SVM Win?

**Setup:** Simulated 100-observation dataset with visible non-linear class separation

| Model | Kernel | Training Error | Test Error |
|-------|--------|---------------|------------|
| Support Vector Classifier | Linear | 12% | 8% |
| **Polynomial SVM** | **Degree 4** | **10%** | **8%** |
| RBF SVM | Radial | 12% | 8% |

> **Finding:** The **polynomial kernel (degree 4) achieves the lowest training error (10%)** — confirming that non-linear kernels better capture the true separation structure in training data. On the test set, all three models tie at **8% error**, showing that for this dataset, the simpler linear SVC generalizes equally well to unseen data despite a higher training error.
>
> **Decision boundary plots** visually confirmed that polynomial and RBF kernels produce curved boundaries closely following the non-linear class structure, while the linear SVC produces a straight separation line.

---

### Exercise 5 — SVM vs Logistic Regression on Quadratic Boundary

**Setup:** n=500, p=2, true boundary: Y = X₁² − X₂² > 0 (quadratic)

| Model | Decision Boundary | Performance |
|-------|------------------|------------|
| Linear Logistic Regression | Straight line | ❌ Fails — misclassifies large portion |
| **Non-linear Logistic Regression** | Curved (engineered features) | ✅ Dramatically improved |
| Linear SVC | Straight line | ❌ Fails on quadratic boundary |
| **Polynomial SVM (degree 2)** | Curved | ✅ **Highest accuracy — best model** |

**Non-linear Logistic Regression features engineered:**
`X₁²`, `X₁ · X₂`, `log|X₂|`

> **Key Insight:** Standard linear logistic regression failed entirely — its decision surface is a straight line, fundamentally unable to recover a quadratic boundary.
>
> **Non-linear logistic regression** with manually engineered quadratic features dramatically improved classification, producing a curved boundary closely matching the true data structure.
>
> **Polynomial SVM achieved the highest accuracy** — demonstrating SVMs' ability to capture complex boundaries **without manual feature construction**, a major practical advantage in production ML systems.

---

### Exercise 6 — Regularization Trade-off: Strict vs Lenient SVM Margin

**Setup:** Simulated barely linearly separable data, C ∈ {0.1, 1, 10, 100, 1000}

| C Value | Training Misclassifications | CV Error | Test Error | Overfitting? |
|---------|---------------------------|---------|-----------|-------------|
| 0.1 | Higher | Higher | Higher | ❌ Underfits |
| **1** | **Moderate** | **Lowest** | **Lowest** | ✅ **Best balance** |
| 10 | Lower | Higher | Higher | ⚠️ Increasing |
| 100 | Near-zero | Higher | Higher | ❌ Overfit |
| 1000 | ~Zero | Highest | Highest | ❌ Severe overfit |

> **Validated Claim:** For barely separable data, **C = 1 minimized both CV and test error** — confirming that tolerating a small number of training misclassifications (soft margin) consistently outperforms forcing a perfect training fit (hard margin) in real-world generalization.
>
> Large C values (100, 1000) achieved near-zero training errors but performed **significantly worse on the test set** — a textbook demonstration of the bias-variance tradeoff. Cross-validation correctly identified C = 1 as optimal before any test data was observed.

---

### Exercise 7 — Auto MPG Classification (High vs Low Fuel Efficiency)

**Setup:** 392 vehicles, binary target: mpg above/below median, features scaled with StandardScaler

**Features used:** `displacement`, `horsepower`, `weight`, `acceleration` (standardized)

#### GridSearchCV Best Parameters:

| Kernel | Best C | Best Other Param | CV Error |
|--------|--------|-----------------|---------|
| **Linear** | **1** | — | **8.66%** ✅ Best |
| Polynomial | 100 | degree = 5 | Higher |
| RBF | 1 | gamma = 0.5 | Comparable |

**Key Results:**
- Feature scaling was **critical** — without StandardScaler, high-magnitude features (weight: 1,600–5,140) dominate over low-magnitude ones (acceleration: 8–25), corrupting distance calculations
- **Linear SVC with C = 1 achieved the lowest CV error of 8.66%** — the preferred production model
- Polynomial and RBF kernels required broader hyperparameter searches but **did not improve over the linear baseline** — the fuel efficiency boundary is approximately linear in the scaled feature space
- Decision boundary plots were generated for each kernel across feature pairs, visualizing how each model separates high- and low-MPG vehicles

> **Business Insight:** For automotive fuel efficiency classification, a simple linear SVM with proper feature scaling achieves ~91% accuracy — sufficient for fleet management, emissions compliance screening, or customer vehicle recommendation systems.

---

### Exercise 8 — Orange Juice Brand Purchase Prediction (CH vs MM)

**Setup:** 800 training / 270 test observations, C ∈ {0.01, 0.1, 1, 10, 100}, 5-fold CV

#### Initial Model (C = 0.01):
- **435 support vectors** required for the linear classifier — indicating a very soft margin with heavy regularization, allowing many training points to influence the boundary

#### After Cross-Validation Tuning:

| Model | Kernel | Initial C | Best C (CV) | Best Test Error |
|-------|--------|-----------|-------------|----------------|
| **Linear SVC** | Linear | 0.01 | CV-tuned | **~18.15% ✅ Best** |
| RBF SVM | Radial | 0.01 | CV-tuned | Comparable |
| Polynomial SVM | Degree 2 | 0.01 | CV-tuned | Comparable |

> **Finding:** After cross-validation tuning, all three kernels converged to **similar test errors**, with the **linear kernel narrowly achieving the best test error of ~18.15%**. This consistency across kernels suggests the OJ purchase decision boundary is **well-approximated by a linear function** of price, loyalty, and discount features.
>
> **Key Insight:** Investing in more complex kernels does not always pay off — for this dataset, the simpler linear model is more efficient and equally accurate. The large number of support vectors (435) at C = 0.01 indicates room for margin tightening — which CV-tuning successfully resolved.

---

## 📊 Overall Model Comparison Summary

| Exercise | Dataset | Best Model | Best Test Error | Key Takeaway |
|----------|---------|-----------|----------------|--------------|
| Ex. 4 | Simulated | All tied | **8%** | Non-linear kernels reduce training error but not always test error |
| Ex. 5 | Simulated | Poly SVM (deg. 2) | Best accuracy | SVM avoids manual feature engineering needed by logistic regression |
| Ex. 6 | Simulated | Linear SVC (C=1) | **Lowest** | CV correctly identifies soft margin as optimal for barely-separable data |
| Ex. 7 | Auto | Linear SVC (C=1) | **8.66% CV error** | Feature scaling is critical; linear boundary sufficient |
| Ex. 8 | OJ | Linear SVC (CV-tuned) | **~18.15%** | Complex kernels add no value when boundary is approximately linear |

---

## 💡 Business Insights

1. **Kernel Selection Should Be Data-Driven:** Polynomial and RBF kernels improve training accuracy but do not always generalize better. Always use cross-validation — not training error — to select the kernel and hyperparameters.

2. **Feature Scaling is Non-Negotiable for SVMs:** On the Auto dataset, applying StandardScaler before fitting reduced CV error significantly. In production ML pipelines, always include scaling in the preprocessing pipeline when using distance-based models.

3. **Soft Margin SVMs Prevent Costly Overfitting:** At C=1000, training error ≈ 0% but test error was the highest — a dangerous pattern in production. The cross-validated soft margin (C=1) generalized reliably, demonstrating why strict perfect-fit models should be avoided in deployed systems.

4. **SVMs Outperform Logistic Regression on Non-Linear Data Without Feature Engineering:** In Exercise 5, SVM with polynomial kernel outperformed manually feature-engineered logistic regression. This translates to reduced development time in NLP, image classification, and fraud detection — where feature engineering is expensive.

5. **Support Vector Count Signals Margin Quality:** 435 support vectors at C=0.01 on OJ data means almost half the training data influences the boundary — an overly soft margin. After CV tuning, fewer support vectors resulted in a tighter, more confident boundary.

---

## 🗂️ File Structure

```
Chapter_9_Applied_Exercise_Solutions/
│
├── Chapter_9.ipynb          ← Main analysis notebook (all exercises)
├── chapter_9.html           ← Rendered HTML version (easy browser viewing)
├── chapter_9.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn pandas numpy matplotlib statsmodels jupyter

# Clone repository
git clone https://github.com/shadalishah/Applied_solutions_of_ISLP.git

# Launch notebook
jupyter notebook Chapter_9.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 9: Support Vector Machines — Applied Exercises 4–8.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository
[ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM)
provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
