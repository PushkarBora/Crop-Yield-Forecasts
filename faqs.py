# faqs.py
"""
FAQ Database for Crop Yield Prediction Platform
Add new questions and answers here - no need to touch app.py!
"""

FAQ_DATABASE = {
    "what_is_lstm": {
        "question": "What is LSTM and when should I use it?",
        "answer": """LSTM (Long Short-Term Memory) is a recurrent neural network designed for sequential data.

**Use LSTM when:**
- You have time-series data with long-term dependencies
- Your data has 50+ time points
- You need to capture patterns over multiple time steps

**Key hyperparameters:**
- Hidden size: 8-16 for small datasets, 32-64 for large
- Layers: 2-3 layers work best
- Dropout: 0.2-0.3 to prevent overfitting

**Performance:** LSTM typically achieves R² > 0.85 on crop yield data."""
    },
    
    "interpret_rmse": {
        "question": "How do I interpret RMSE values?",
        "answer": """RMSE (Root Mean Square Error) measures prediction accuracy in the same units as your target variable.

**For crop yield prediction:**
- RMSE < 5 kg/ha: Excellent
- RMSE 5-10 kg/ha: Good
- RMSE 10-20 kg/ha: Acceptable
- RMSE > 20 kg/ha: Needs improvement

**Lower RMSE = Better predictions**

Compare RMSE across models to find the best performer."""
    },
    
    "choose_model": {
        "question": "Which model should I choose?",
        "answer": """**Model Selection Guide:**

🥇 **LSTM/GRU:** Best for long sequences (50+ points), captures complex patterns
📊 **Transformer:** Best for very large datasets (500+ points), parallel training
🌲 **Random Forest:** Best for small datasets (<100 points), easy to interpret
📈 **ARIMA:** Best for univariate data, classical statistics
🔄 **RNN:** Good for short sequences, faster than LSTM
🧠 **ANN:** Good for tabular data, simple relationships

**Recommendation:** Start with LSTM or Random Forest, they work well for most crop yield tasks."""
    },
    
    "hyperparameter_tuning": {
        "question": "What is hyperparameter tuning?",
        "answer": """Hyperparameter tuning finds the best settings for your model.

**Auto-tune checkbox:**
✅ **Enabled:** System tests multiple combinations (slower but better)
❌ **Disabled:** Uses your manual settings (faster but may not be optimal)

**Common hyperparameters:**
- **Learning rate:** How fast the model learns (0.001-0.01)
- **Dropout:** Prevents overfitting (0.2-0.4)
- **Epochs:** Training iterations (100-300)
- **Hidden size:** Model capacity (8-64)

**Tip:** Enable auto-tune for best results, especially on new datasets!"""
    },
    
    "r2_meaning": {
        "question": "What does R² score mean?",
        "answer": """R² (R-squared) measures how well your model explains the data variance.

**Scale:** 0 to 1 (or 0% to 100%)

**Interpretation:**
- R² > 0.9: Excellent fit
- R² = 0.7-0.9: Good fit
- R² = 0.5-0.7: Moderate fit
- R² < 0.5: Poor fit

**Example:** R² = 0.85 means your model explains 85% of yield variation.

**Higher R² = Better predictions**"""
    },
    
    "multivariate_vs_univariate": {
        "question": "Multivariate vs Univariate models?",
        "answer": """**Univariate (ARIMA only):**
- Uses only past yield values
- Simpler, faster
- Good for stable trends
- Example: Predict yield using only historical yields

**Multivariate (LSTM, GRU, RNN, etc):**
- Uses yield + weather (rain, temperature)
- More accurate
- Captures complex relationships
- Example: Predict yield using yields, rainfall, temperature

**Recommendation:** Use multivariate models when you have weather data - they're much more accurate!"""
    },
    
    "overfitting": {
        "question": "How do I prevent overfitting?",
        "answer": """Overfitting = model memorizes training data but fails on new data.

**Signs of overfitting:**
- Train R² much higher than Test R²
- Example: Train R² = 0.95, Test R² = 0.60

**Prevention strategies:**
1. **Increase dropout:** Try 0.3-0.4
2. **Reduce model size:** Use fewer layers/hidden units
3. **Early stopping:** Enable patience parameter
4. **More data:** Add more training samples
5. **Data augmentation:** Use cross-validation

**Tip:** A small gap between train/test R² (0.05-0.1) is normal!"""
    },
    
    "forecast_inputs": {
        "question": "What weather inputs do I need for forecasting?",
        "answer": """For future predictions, you need weather forecasts for each prediction period.

**Required inputs:**
1. **Rainfall (mm):** Expected precipitation
2. **Temperature (°C):** Mean temperature

**Where to get forecasts:**
- Weather stations
- Climate models
- Historical averages (use past data as estimate)

**Example for 5-year horizon:**
Year 1: 45mm rain, 28°C
Year 2: 50mm rain, 27°C
...and so on

**Tip:** Use historical averages if you don't have forecasts!"""
    },
    
    "mape_meaning": {
        "question": "What is MAPE and how do I use it?",
        "answer": """MAPE (Mean Absolute Percentage Error) shows prediction error as a percentage.

**Interpretation:**
- MAPE < 5%: Excellent accuracy
- MAPE 5-10%: Good accuracy
- MAPE 10-20%: Acceptable accuracy
- MAPE > 20%: Poor accuracy

**Example:** MAPE = 8% means predictions are off by 8% on average.

**Lower MAPE = Better predictions**

**Advantage:** Easy to understand - anyone can grasp "8% error"!"""
    },
    
    "data_preparation": {
        "question": "How should I prepare my dataset?",
        "answer": """**Required columns:**
- `Year` or `Date`: Time column
- `Yield`: Target variable (kg/ha)
- `rain`: Rainfall (mm) - optional
- `mean_t`: Temperature (°C) - optional

**Data requirements:**
- Minimum 20 rows (more is better)
- No missing values
- Consistent time intervals

**File formats:**
- CSV (.csv) - recommended
- Excel (.xlsx)

**Example:**
Year,Yield,rain,mean_t
2010,2500,450,26.5
2011,2650,480,27.0

**Tip:** More data = better predictions!"""
    },
    
    "train_test_split": {
        "question": "What is train/test split?",
        "answer": """Train/test split divides your data into two parts:

**Training set (default 85%):**
- Model learns patterns from this data
- Used to fit the model

**Test set (default 15%):**
- Model never sees this during training
- Used to evaluate real-world performance

**Example with 100 data points:**
- Train: First 85 points (1985-2069)
- Test: Last 15 points (2070-2084)

**Why it matters:**
Test performance shows how well your model will work on future, unseen data.

**Tip:** Use default 85/15 split - it works well for most datasets!"""
    },
    
    "comparison_mode": {
        "question": "How does model comparison work?",
        "answer": """Model comparison trains multiple models and shows you which performs best.

**How to use:**
1. Select 2+ models from the list
2. Upload your dataset
3. Configure hyperparameters
4. Click "Train Models"
5. Compare results side-by-side

**Comparison metrics:**
- RMSE (lower = better)
- R² (higher = better)
- MAPE (lower = better)

**Best model:**
Automatically highlighted based on lowest RMSE

**Tip:** Compare LSTM, GRU, and Random Forest - they usually perform best!"""
    }
}


def get_all_faqs():
    """Return all FAQs as a list"""
    return [
        {"key": key, "question": value["question"]}
        for key, value in FAQ_DATABASE.items()
    ]


def get_faq_answer(key):
    """Get answer for a specific FAQ key"""
    if key in FAQ_DATABASE:
        return FAQ_DATABASE[key]
    return None