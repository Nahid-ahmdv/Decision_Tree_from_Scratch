# **Decision Trees for Classification & Regression (with a From-Scratch Implementation)**

## **Overview**
This project is a guided walkthrough of **decision trees**, with the main goal of **building a working decision tree from scratch** and validating that it behaves like a real machine-learning model. Along the way, the notebook walks through a few focused experiments—first to build intuition, then to test the custom implementation on both **classification** and **regression** problems.

The notebook unfolds in three main steps:

1. **Toy regression (intuition first):** we start with a tiny 1D dataset to see how a tree creates **piecewise-constant** predictions by splitting the input space.
2. **Real-world classification:** we train and tune a scikit-learn tree to predict **cardiovascular disease** using a dataset of **70,000 patients**.
3. **From-scratch decision tree:** we implement the full algorithm (splits, recursion, stopping rules, prediction, and `predict_proba`) and validate it on synthetic and benchmark datasets.


## **Datasets**

### **1) Cardiovascular Disease Dataset (main real-world example)**
The notebook uses `mlbootcamp5_train.csv` (70,000 patients), commonly shared through the mlcourse.ai materials.

- **Source:** `https://github.com/Yorko/mlcourse.ai/blob/master/data/mlbootcamp5_train.csv`
- **Target:** `cardio` (0/1 — absence/presence of cardiovascular disease)

**Feature summary (as used in the notebook):**
- Objective: `age` (days), `height` (cm), `weight` (kg), `gender`
- Examination: `ap_hi` (systolic BP), `ap_lo` (diastolic BP), `cholesterol` (1/2/3), `gluc` (1/2/3)
- Subjective: `smoke`, `alco`, `active` (binary)

### **2) Benchmark / synthetic datasets (used to validate the scratch model)**
- `make_classification` (synthetic classification sanity check)
- `load_digits` (handwritten digits classification)
- `make_regression` (synthetic 1D regression sanity check)
- `fetch_california_housing` (regression benchmark; the notebook subsamples 5,000 examples for speed)


## **Methodology & Workflow**

### **1) Toy regression: how splits work**
We begin with a tiny 1D dataset generated from a known function (`y = x³`) to make the mechanics visible.  
The notebook computes split quality using a **variance-based criterion** and shows how the “best threshold” produces a cleaner partition of the target values.

### **2) Heart disease prediction with scikit-learn (baseline + tuning)**
Next, we move to the cardiovascular disease dataset and train a baseline `DecisionTreeClassifier`.

**Preprocessing**
- Convert `age` from days to years: `age_years = floor(age / 365.25)`
- One-hot encode `cholesterol` and `gluc` (three binary indicators each)

**Training + evaluation**
- Stratified train/holdout split
- Hyperparameter tuning with `GridSearchCV` over `max_depth` (2 → 10)
- 5-fold `StratifiedKFold` cross-validation
- Metric: **accuracy**

### **3) Feature engineering inspired by SCORE (interpretability angle)**
To connect the model to clinical reasoning, the notebook builds binary indicators inspired by SCORE-style risk categories (age bins, SBP bins, cholesterol categories, sex, smoking).  
The resulting tree highlights **systolic blood pressure** very early, reinforcing how trees can produce **interpretable rules** aligned with domain expectations.

### **4) Decision tree from scratch (core contribution)**
The main contribution is a custom `DecisionTree` implementation built around:
- A `Node` structure (feature index, threshold, left/right children, or leaf labels)
- A greedy split search over **unique thresholds** per feature
- Recursive growth with stopping rules:
  - `max_depth`
  - `min_samples_split`

**Supported split criteria**
- Classification: `gini`, `entropy`
- Regression: `variance`, `mad_median` (mean absolute deviation from the median)

**Leaf predictions**
- Classification leaf: majority class (`np.bincount(y).argmax()`)
- Regression leaf: mean target value (`np.mean(y)`)

**Extras**
- `predict_proba()` for classification
- scikit-learn compatibility via `BaseEstimator` + parameter handling (so it works with `GridSearchCV`)


## **Key Results & Insights**

### **Heart disease (scikit-learn tree)**
- Baseline holdout accuracy (max_depth=3): **0.7260**
- Best CV mean accuracy (selected model): **0.7294 ± 0.0021**
- Tuned holdout accuracy (best `max_depth=4`): **0.7299**
- Relative improvement from tuning: **~0.54%**

**Interpretation:** depth tuning helps, but only modestly for a single tree still, the result is strong for an interpretable baseline.

### **Scratch tree: classification validation**
- Synthetic classification (custom tree): **Accuracy = 0.85** (and `predict_proba` verified)

### **Scratch tree: digits benchmark (criterion comparison + tuning)**
Using `GridSearchCV` over `max_depth` (3 → 12):
- **GINI:** best `max_depth=10`, best CV accuracy **0.8476**
- **ENTROPY:** best `max_depth=10`, best CV accuracy **0.8524**
- Final entropy model test accuracy: **0.8833**

**Interpretation:** entropy slightly outperforms Gini here, and depth selection matters.

### **Scratch tree: regression validation**
- Synthetic 1D regression (custom tree, `mad_median`, depth=6): **Test MSE = 165.7607**  
  (used mainly as a sanity check + visualization of piecewise predictions)

### **Scratch tree: California Housing (criterion comparison + tuning)**
On a 5,000-sample subset:
- Depth=2 comparison:
  - `variance`: **Test MSE = 0.9367**
  - `mad_median`: **Test MSE = 0.9546**
- Cross-validating `max_depth` (2 → 8):
  - `variance`: best CV MSE **0.5559** at `max_depth=7`
  - `mad_median`: best CV MSE **0.5353** at `max_depth=6`

**Interpretation:** the “best” regression criterion depends on depth `mad_median` wins after tuning even though it loses at depth 2.


## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/Nahid-ahmdv/Decision_Tree_from_Scratch.git
   cd Decision_Tree_from_Scratch
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

From there, you can follow along with the codes.


## **Conclusion**
This project shows decision trees from two angles: as a practical, interpretable model on real data and as an algorithm you can build yourself. The from-scratch implementation is the centerpiece: it demonstrates the full split-selection and recursion logic, supports multiple criteria for both tasks, and holds up on synthetic checks and benchmark datasets. The experiments also make one theme very clear: **tree depth and split criterion shape both performance and the story the model tells**.
