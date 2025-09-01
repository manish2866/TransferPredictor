# ⚽ FIFA Players Analysis & Prediction Web App  

## 📌 Overview  
This project is a **Streamlit-based web application** that allows users to:  
- Upload and clean FIFA player datasets.  
- Perform **Exploratory Data Analysis (EDA)** and **visualizations**.  
- Train and test ML models for player performance prediction.  
- Input custom player data and predict outcomes using the trained model.  

## ⚙️ Features  

- **Data Cleaning & Preprocessing**  
  - Notebook (`Data cleaning and EDA.ipynb`) performs feature engineering & prepares datasets.  

- **Machine Learning Models**  
  - `models.ipynb` trains & evaluates models on FIFA player stats.  

- **Interactive Web App (Streamlit)**  
  - Upload a CSV file with required fields.  
  - Automatic preprocessing and feature engineering.  
  - Visualization of key statistics.  
  - Predictions based on trained model.  

- **Visualizations**  
  - Implemented in `Visualizations.py`.  
  - Includes distribution plots, comparisons, and feature importance.  

---

## 🛠 Setup Instructions  

### 1. Clone the Repository  
```bash
git clone <your-repo-url>
cd FIFA_Project
```

### 2. Install Dependencies  
```bash
pip install streamlit pandas numpy matplotlib scikit-learn seaborn
```

### 3. Run the Application  
```bash
streamlit run app.py
```

By default, the app will open in your browser at:  
👉 `http://localhost:8501/`  

---

## 🚀 Usage  

1. Upload a clean CSV file with required fields (player stats).  
2. The app will:  
   - Perform feature engineering.  
   - Train/test models.  
   - Generate visualizations.  
3. Enter custom player attributes to get predictions.  

---

## 📊 Example Dataset Fields  

The input CSV must include the following fields【34†source】:  

```
['name', 'full_name', 'birth_date', 'age', 'height_cm', 'weight_kgs',
 'positions', 'nationality', 'overall_rating', 'potential', 'value_euro',
 'wage_euro', 'preferred_foot', 'international_reputation(1-5)',
 'weak_foot(1-5)', 'skill_moves(1-5)', 'body_type',
 'release_clause_euro', 'crossing', 'finishing', 'heading_accuracy',
 'short_passing', 'volleys', 'dribbling', 'curve', 'freekick_accuracy',
 'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
 'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
 'vision', 'penalties', 'composure', 'marking', 'standing_tackle',
 'sliding_tackle']
```

---

## 📝 Notes  

- Make sure the dataset matches the expected schema and datatypes.  
- Models are trained dynamically each run but can be extended to use pre-trained models.  
- Visualizations provide insights into player stats distribution and model performance.  
