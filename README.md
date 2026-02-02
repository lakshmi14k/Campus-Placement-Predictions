 **Campus Placement Prediction System :** Predictive Analytics for Student Placement Outcomes

[Live Demo](https://campus-placements-hy5puv27opikguym8l7ihq.streamlit.app/) | Machine Learning | Streamlit Application

**Problem Statement:**

Educational institutions and students need data-driven insights to understand placement likelihood and improve career outcomes. Traditional placement processes lack predictive capabilities, making it difficult to identify at-risk students or optimize career preparation strategies.

**Challenges:**
- No visibility into factors influencing placement success
- Students lack personalized guidance on improving placement chances
- Career services teams cannot proactively target interventions
- Limited understanding of how academic performance correlates with hiring outcomes

**Solution Overview:**

An end-to-end machine learning application that predicts student placement probability based on academic performance, demographics, and experience factors. The system provides:

- **Predictive Analytics:** Logistic regression model forecasting placement outcomes
- **Interactive Interface:** Streamlit web application for real-time predictions
- **Exploratory Insights:** Visual analytics on placement patterns by demographics and specialization
- **Self-Healing Architecture:** Continuous model retraining as new data is contributed

**Dataset Overview:**

**Source:** Campus placement records (anonymized student data)

**Records:** 215 students with placement outcomes

**Features (15 columns):**
- **Academic Performance:** SSC %, HSC %, Degree %, MBA %
- **Educational Background:** SSC board, HSC board, degree type, specialization
- **Experience:** Work experience (Yes/No)
- **Aptitude:** Entrance test score
- **Demographics:** Gender
- **Target Variable:** Placement status (Placed/Not Placed)

**Data Characteristics:**
- Clean dataset with minimal missing values
- Balanced gender distribution
- Multiple specializations (Finance, Marketing, HR)
- Real student outcomes from placement season

**Tech Stack:**

| Component | Technology |
|--|--|
| **Data Analysis** | Python (pandas, numpy, matplotlib, seaborn, plotly) |
| **Machine Learning** | scikit-learn (Logistic Regression, GridSearchCV) |
| **Model Persistence** | pickle (model serialization) |
| **Frontend** | Streamlit (interactive web application) |
| **Deployment** | Streamlit Community Cloud |
| **Development** | Jupyter Notebook (EDA), Python IDE |

** Project Structure:**

```
Campus-Placement-Prediction-ML/
├── data/
│   └── Placement_Data_Full_Class.csv     Student placement dataset
├── notebooks/
│   └── placements_EDA.ipynb              Exploratory data analysis
├── app/
│   ├── streamlit-machine.py              Streamlit application
│   ├── trained_model.sav                 Serialized ML model
│   └── requirements.txt                  Python dependencies
├── assets/
│   ├── campus-placements.jpg             Application banner
│   ├── image-1.jpg                       UI screenshot 1
│   ├── image-2.png                       UI screenshot 2
│   └── image-3.jpg                       Analytics visualization
├── docs/
│   └── ARCHITECTURE.md                   System architecture documentation
└── README.md
```

**System Architecture:**

 Self-Healing ML Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERACTION                      │
│  (Streamlit Interface - Input academic/demographic data) │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              PREDICTION PIPELINE                         │
│  1. Preprocess Input (encode categoricals)              │
│  2. Load Trained Model (pickle)                          │
│  3. Generate Prediction (placement probability)          │
│  4. Display Result (placed/not placed + confidence)      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           OPTIONAL: DATA CONTRIBUTION                    │
│  User can submit their actual outcome to dataset         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            SELF-HEALING MECHANISM                        │
│  1. Append new data to CSV                               │
│  2. Retrain model (GridSearchCV hyperparameter tuning)   │
│  3. Save updated model                                   │
│  4. Refresh analytics dashboard                          │
└─────────────────────────────────────────────────────────┘
```

**Key Innovation:** Continuous learning loop where user contributions automatically trigger model retraining, keeping predictions current with latest placement trends.

**Methodology:**

 1. Exploratory Data Analysis

**Key Findings from EDA:**
- Gender distribution analysis revealed placement patterns across demographics
- Specialization (Finance vs. Marketing vs. HR) showed varying placement rates
- Academic percentages (SSC, HSC, Degree, MBA) correlated with placement success
- Work experience emerged as significant predictor of placement outcomes
- Entrance test scores showed moderate correlation with final placement

**Visualizations Created:**
- Gender distribution pie charts
- Placement status by specialization (count plots)
- Academic performance distributions (histograms)
- Correlation heatmaps for numeric features

 2. Machine Learning Model

**Model Selection:** Logistic Regression
- **Why:** Interpretable coefficients showing feature importance
- **Performance:** Binary classification (Placed vs. Not Placed)
- **Optimization:** GridSearchCV hyperparameter tuning

**Hyperparameter Search:**
```python
parameters = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [1, 2, 3, 5, 10, 20, 30, 50],
    'max_iter': [100, 200, 300]
}
```

**Features Used:**
- Gender (encoded)
- SSC percentage and board
- HSC percentage, board, and stream
- Degree percentage and type
- Work experience
- Entrance test percentage
- MBA specialization and percentage

 3. Streamlit Application Features

**Prediction Interface:**
- User input forms for all 15 features
- Real-time prediction on button click
- Confidence score display
- Clear placed/not placed result

**Analytics Dashboard:**
- Gender distribution visualizations
- Placement rates by specialization
- Popular degree majors analysis
- Interactive plots using Plotly

**Data Contribution:**
- Users can submit their actual placement outcomes
- Triggers automatic model retraining
- Updates analytics in real-time
- Implements continuous learning feedback loop

**Installation & Usage:**

 Local Setup

**Prerequisites:**
- Python 3.8+
- pip package manager

**Steps:**

1. **Clone Repository**
```bash
git clone https://github.com/lakshmi14k/Campus-Placement-Prediction-ML.git
cd Campus-Placement-Prediction-ML
```

2. **Install Dependencies**
```bash
pip install -r app/requirements.txt
```

3. **Update File Paths**
Edit `app/streamlit-machine.py`:
```python
 Change common_path to your local directory
common_path = "/path/to/your/Campus-Placement-Prediction-ML"
```

4. **Run Application**
```bash
streamlit run app/streamlit-machine.py
```

Application opens in browser at `http://localhost:8501`

**Live Demo:**

Access the deployed application: [Campus Placement Predictor](https://campus-placements-hy5puv27opikguym8l7ihq.streamlit.app/)

**Results & Performance:**

 Model Performance
- **Algorithm:** Logistic Regression with GridSearchCV optimization
- **Training Approach:** 80-20 train-test split, 10-fold cross-validation
- **Optimization:** Hyperparameter tuning across penalty types and regularization strength

 Key Insights Discovered

**High-Impact Features:**
- Work experience significantly increases placement probability
- MBA percentage and entrance test scores show strong correlation
- Specialization choice influences outcomes (Finance > Marketing > HR in sample data)
- Academic consistency across SSC/HSC/Degree matters more than individual scores

**Demographic Patterns:**
- Gender distribution relatively balanced in placements
- STEM degrees show different placement patterns than Commerce/Arts
- Board of education (Central vs. State) has minimal impact on placement

**Actionable Recommendations:**
- Students should prioritize internships/work experience
- Maintain consistent academic performance across all levels
- Entrance test preparation is crucial for competitive placements
- Specialization choice should align with market demand

Academic Context

**Course:** INFO 6105 - Data Science Engineering Methods and Tools  
**Institution:** Northeastern University  
**Project Type:** Collaborative academic capstone  
**Semester:** Spring 2024

This project was completed as part of an academic curriculum to demonstrate end-to-end ML application development, including data analysis, model training, hyperparameter optimization, and production deployment.

 🔄 Self-Healing Architecture

**Continuous Learning Design:**

The application implements a feedback loop where user contributions automatically improve the model:

1. **User submits prediction request** → Gets placement probability
2. **User contributes actual outcome** → Data appended to CSV
3. **System detects new data** → Triggers `rerun_model()`
4. **Model retraining** → GridSearchCV with updated dataset
5. **Model replacement** → New model saved, analytics refreshed
6. **Updated predictions** → Future users benefit from improved model

**Benefits:**
- Model stays current with latest placement trends
- Handles data drift as hiring patterns change
- Community-driven model improvement
- No manual retraining required

**Technical Implementation:**

 Core Functions

**1. Prediction Pipeline**
```python
def enhance_input(gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, 
                  degree_p, degree_t, workex, etest_p, 
                  specialisation, mba_p):
     Encode categorical variables
     Structure input vector
     Load trained model
     Generate prediction
     Return result with confidence
```

**2. Model Retraining**
```python
def rerun_model():
     Reload full dataset (including new contributions)
     Encode categorical features
     Split train/test
     Initialize Logistic Regression
     GridSearchCV hyperparameter tuning
     Save updated model
```

**3. Analytics Dashboard**
```python
def run_analytics():
     Load dataset
     Generate visualizations:
       - Gender distribution (pie chart)
       - Placement by gender (count plot)
       - Specialization ratios
       - Popular majors analysis
```

Sample Predictions

**Example 1: High Placement Probability**
- Gender: Male
- SSC: 85%, HSC: 82%, Degree: 78%, MBA: 75%
- Work Experience: Yes
- Entrance Test: 88%
- **Prediction:** PLACED (92% confidence)

**Example 2: Moderate Placement Probability**
- Gender: Female
- SSC: 65%, HSC: 68%, Degree: 70%, MBA: 72%
- Work Experience: No
- Entrance Test: 65%
- **Prediction:** NOT PLACED (68% confidence) - *Recommendation: Gain work experience*
**
Limitations & Disclaimers:**

**Model Limitations:**
- Trained on limited dataset (215 students from specific institution)
- May not generalize to all universities or geographic regions
- Predictions are probabilistic, not guaranteed outcomes
- Historical data may not reflect current job market conditions

**Ethical Considerations:**
- Predictions should inform preparation, not discourage students
- Tool is advisory only, not deterministic career guidance
- Individual effort and circumstances matter beyond model features
- Should be used alongside career counseling, not as replacement

**Academic Project:**
- Built for educational purposes demonstrating ML application development
- Not intended for production use without validation on broader datasets
- Demonstrates technical capabilities in data science and software engineering

**Resources:**

- **[Live Application](https://campus-placements-hy5puv27opikguym8l7ihq.streamlit.app/)** - Try the predictor
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Technical deep-dive

*Built with Python, scikit-learn, and Streamlit | Deployed on Streamlit Community Cloud*
