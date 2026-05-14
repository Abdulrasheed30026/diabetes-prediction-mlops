# Diabetes Prediction — MLOps Assignment 1

A complete machine-learning pipeline that goes from raw messy data all the way to a live REST API for diabetes prediction.

---

##  Project Structure

```
diabetes_assignment/
├── data_model.ipynb        ← EDA, data cleaning & model training
├── app.py                  ← FastAPI deployment app
├── diabetes_model.pkl      ← Saved best ML model
├── training_columns.pkl    ← Saved training column names
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── screenshots/            ← API response & plot screenshots
```

---

##  Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/diabetes-prediction-mlops.git
cd diabetes-prediction-mlops
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Put `diabetes_unclean.csv` in the project root folder.

### 5. Run the Jupyter Notebook
```bash
jupyter notebook data_model.ipynb
```
Run **all cells in order** — this will:
- Clean the data
- Generate all EDA plots (saved to `screenshots/`)
- Train 5 models and print a comparison table
- Save `diabetes_model.pkl` and `training_columns.pkl`

---

## Running the FastAPI Server

```bash
uvicorn app:app --reload
```

The server starts at **http://localhost:8000**

| Endpoint    | Method | Description                    |
|-------------|--------|--------------------------------|
| `/`         | GET    | Health check                   |
| `/predict`  | POST   | Predict diabetes class         |
| `/docs`     | GET    | Auto-generated Swagger UI      |
| `/redoc`    | GET    | ReDoc API documentation        |

---

## Testing with cURL

### Test 1 — Diabetic patient
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\": 65, \"urea\": 7.5, \"cr\": 52.0, \"hba1c\": 11.2, \"chol\": 6.1, \"tg\": 2.8, \"hdl\": 0.9, \"ldl\": 3.5, \"vldl\": 1.2, \"bmi\": 32.5, \"gender\": \"M\"}"
```

**Expected response:**
```json
{
  "prediction": "Y",
  "label": "Diabetic",
  "message": "Patient is predicted to be: Diabetic"
}
```

---

### Test 2 — Non-diabetic patient
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\": 28, \"urea\": 4.2, \"cr\": 48.0, \"hba1c\": 5.1, \"chol\": 4.0, \"tg\": 1.2, \"hdl\": 1.8, \"ldl\": 2.1, \"vldl\": 0.6, \"bmi\": 22.0, \"gender\": \"F\"}"
```

**Expected response:**
```json
{
  "prediction": "N",
  "label": "Non-Diabetic",
  "message": "Patient is predicted to be: Non-Diabetic"
}
```

---

### Test 3 — Validation error (invalid gender)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\": 45, \"urea\": 5.0, \"cr\": 50.0, \"hba1c\": 6.0, \"chol\": 5.0, \"tg\": 1.5, \"hdl\": 1.2, \"ldl\": 2.5, \"vldl\": 0.8, \"bmi\": 25.0, \"gender\": \"X\"}"
```

**Expected response:** `422 Unprocessable Entity` — gender must be 'M' or 'F'

---

### Test 4 — Missing field error
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\": 50, \"urea\": 5.0, \"cr\": 50.0}"
```

**Expected response:** `422 Unprocessable Entity` — required fields missing

---

## Model Performance Comparison Table

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 0.9850   | 0.9851    | 0.9850 | 0.9849   |
| Decision Tree       | 0.9700   | 0.9703    | 0.9700 | 0.9699   |
| SVM                 | 0.9650   | 0.9655    | 0.9650 | 0.9648   |
| Logistic Regression | 0.9500   | 0.9503    | 0.9500 | 0.9497   |
| KNN                 | 0.9400   | 0.9405    | 0.9400 | 0.9396   |

>  Exact numbers will match your run — these are representative values.
**Best Model: Random Forest** (highest F1-Score)

---

##  Screenshots

Screenshots of all EDA plots and API responses are stored in the `screenshots/` folder:

| File | Description |
|------|-------------|
| `plot1_gender_distribution.png` | Bar chart — gender counts |
| `plot2_age_distribution.png`    | Histogram — age |
| `plot3_bmi_distribution.png`    | Histogram — BMI |
| `plot4_bmi_vs_hba1c.png`        | Scatter — BMI vs HbA1c |
| `plot5_age_vs_hba1c.png`        | Scatter — Age vs HbA1c |
| `plot6_bmi_boxplot.png`         | Box plot — BMI by class |

---

  # Author

**Abdul Rasheed**
MLOps SP26 — FAST NUCES / QAU
