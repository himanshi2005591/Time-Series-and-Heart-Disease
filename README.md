# 🧠📊 Data Science Projects: Time Series Forecasting & Heart Disease Prediction

---

## 📈 Project 1: Time Series Sales Forecasting Using ARIMA

### 📌 Objective
Analyze historical sales data to identify patterns and forecast future sales using ARIMA modeling.

---

### 🗂 Dataset Structure

| Column | Description              |
|--------|--------------------------|
| Date   | Timestamp of the sale    |
| Sales  | Number of units sold     |

---

### ✅ Project Steps

1. **Data Loading & Preprocessing**
   - Read CSV file using `pandas`.
   - Converted `Date` to datetime and set as index.
   - Resampled to monthly data.

2. **Visualization**
   - Line plot of sales to reveal trends and seasonality.
   - Applied moving average smoothing for better trend visibility.

3. **Forecasting Using ARIMA**
   - Built ARIMA model with suitable `(p, d, q)` values.
   - Split into training and testing sets.
   - Forecasted future sales using `statsmodels`.

4. **Evaluation**
   - Compared predicted vs actual values.
   - Used RMSE and MAPE metrics to assess model performance.

---

### 📉 Output

- Forecasted next few months' sales.
- Visualized:
  - Original vs Predicted sales
  - Trend lines using moving averages

---

### 🛠 Tools Used

- Python 3
- `pandas`, `matplotlib`, `seaborn`
- `statsmodels`
- `scikit-learn` (for evaluation metrics)

---

## ❤️ Project 2: Heart Disease Prediction Using Logistic Regression

### 📌 Objective
Predict whether a patient has heart disease based on medical indicators using logistic regression.

---

### 🗂 Dataset Structure

| Column         | Description                              |
|----------------|------------------------------------------|
| Age            | Patient's age                            |
| Gender         | Male or Female                           |
| Cholesterol    | Serum cholesterol level                  |
| Blood Pressure | Systolic blood pressure                  |
| Heart Disease  | 1 = Has disease, 0 = No disease (target) |

---

### ✅ Project Steps

1. **Load & Clean Data**
   - Loaded `heart_disease.csv`.
   - Checked for missing values and removed duplicates.

2. **Feature Engineering**
   - Encoded `Gender` into numeric.
   - Normalized features using `StandardScaler`.

3. **Model Training**
   - Split dataset into train/test sets.
   - Trained a logistic regression classifier.

4. **Model Evaluation**
   - Generated predictions on test set.
   - Evaluated using:
     - Confusion Matrix
     - Accuracy, Precision, Recall, F1-score

---

### 📊 Output

- Binary classification results (heart disease or not).
- Model performed well with appropriate scaling and encoding.
- Classification report generated for performance analysis.

---

### 🛠 Tools Used

- Python 3
- `pandas`, `numpy`
- `scikit-learn` (`LogisticRegression`, `StandardScaler`, metrics)
- `matplotlib`, `seaborn`

---

## ✅ How to Run

1. Clone the repository or download the `.ipynb` files.
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn statsmodels
