import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
from datetime import datetime

# ------------------ Helper Function for Safe Label Encoding ------------------ #
def safe_label_transform(series, le):
    """
    Converts the series to string and replaces any unseen value with the default (first class).
    Then applies the label encoder's transform.
    """
    series = series.astype(str)
    safe_series = series.apply(lambda x: x if x in le.classes_ else le.classes_[0])
    return le.transform(safe_series)

# ========================== PART 1: Model Training & Evaluation ========================== #

# Create output directory for models and outputs
output_dir = r"C:\Users\sudar\Downloads\pythonhackathon\cibilscoremodel\financial_wellbeing_data_full_proper"
os.makedirs(output_dir, exist_ok=True)

# Load dataset (update the path as needed for your environment)
data_path = r"C:\Users\sudar\Downloads\pythonhackathon\cibilscoremodel\financial_wellbeing_data_full_proper.csv"
df = pd.read_csv(data_path)

def preprocess_data(df):
    """
    Preprocess the financial data and generate required features.
    Expected columns: 'Monthly Income', 'Savings & Investments', 'Fixed Expenses',
    'Variable Expenses', 'Existing Loans & Liabilities'
    """
    df_processed = df.copy()
    
    # Calculate Debt-to-Income Ratio if not present
    if 'Debt-to-Income Ratio' not in df_processed.columns:
        df_processed['Debt-to-Income Ratio'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / df_processed['Monthly Income'], 1.0)
    
    # Income Stability Score: Savings & Investments / Monthly Income (capped at 1)
    if 'Income Stability Score' not in df_processed.columns:
        df_processed['Income Stability Score'] = np.minimum(
            df_processed['Savings & Investments'] / df_processed['Monthly Income'], 1.0)
    
    # Credit Utilization Rate: Existing Loans & Liabilities / Monthly Income (capped at 1)
    if 'Credit Utilization Rate' not in df_processed.columns:
        df_processed['Credit Utilization Rate'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / df_processed['Monthly Income'], 1.0)
    
    # Monthly Savings Percentage: (Monthly Income - (Fixed Expenses + Variable Expenses)) / Monthly Income (min 0)
    if 'Monthly Savings Percentage' not in df_processed.columns:
        df_processed['Monthly Savings Percentage'] = np.maximum(
            (df_processed['Monthly Income'] - (df_processed['Fixed Expenses'] + df_processed['Variable Expenses']))
            / df_processed['Monthly Income'], 0)
    
    # Financial Volatility Index: default value 0.5 (since 'Monthly Cash Flow Trends' is not available)
    if 'Financial Volatility Index' not in df_processed.columns:
        df_processed['Financial Volatility Index'] = 0.5
    
    # Financial Resilience Score: Savings & Investments / (Fixed Expenses + Variable Expenses + 1), capped at 12
    if 'Financial Resilience Score' not in df_processed.columns:
        df_processed['Financial Resilience Score'] = np.minimum(
            df_processed['Savings & Investments'] / (df_processed['Fixed Expenses'] + df_processed['Variable Expenses'] + 1), 12)
    
    # Financial Stress Index: Existing Loans & Liabilities / (Savings & Investments + 1), capped at 5
    if 'Financial Stress Index' not in df_processed.columns:
        df_processed['Financial Stress Index'] = np.minimum(
            df_processed['Existing Loans & Liabilities'] / (df_processed['Savings & Investments'] + 1), 5)
    
    # Generate target Financial Well-being Score if not present
    if 'Financial Well-being Score' not in df_processed.columns:
        df_processed['Financial Well-being Score'] = (
            df_processed['Income Stability Score'].fillna(0.5) * 0.35 +
            (1 - df_processed['Credit Utilization Rate'].fillna(0.5)) * 0.25 +
            df_processed['Monthly Savings Percentage'].fillna(0.5) * 0.25 +
            (1 - df_processed['Financial Volatility Index'].fillna(0.5)) * 0.15
        ).round(2)
    
    return df_processed

# Preprocess data
df_processed = preprocess_data(df)

# Compute correlation matrix (for reference)
correlation = df_processed.select_dtypes(include=[np.number]).corr()

# --- Define Features and Target ---
categorical_columns = ['Spending Behavior', 'Investment Strategy']
numerical_features = [
    'Income Stability Score', 'Debt-to-Income Ratio', 'Credit Utilization Rate',
    'Monthly Savings Percentage', 'Financial Volatility Index', 
    'Financial Resilience Score', 'Financial Stress Index'
]
target = 'Financial Well-being Score'

# Encode categorical features (convert to string first)
label_encoders = {}
for col in categorical_columns:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = df_processed[col].astype(str)
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

# Split data into training and testing sets
X = df_processed[numerical_features + categorical_columns]
y = df_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ],
    remainder='passthrough'
)

# --- Model Training and Selection ---
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'ElasticNet': ElasticNet(random_state=42)
}

param_grids = {
    'RandomForest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5]
    },
    'ElasticNet': {
        'model__alpha': [0.1, 0.5, 1.0],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    }
}

best_model = None
best_score = -float('inf')
best_model_name = None
model_results = {}

for name, model in models.items():
    # Removed printing for training progress and metrics
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    grid_search = GridSearchCV(
        pipeline, 
        param_grids[name], 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    tolerance = 0.05  # 5% tolerance
    accuracy = np.mean(np.abs(y_pred - y_test) <= tolerance) * 100
    
    model_results[name] = {
        'pipeline': best_pipeline,
        'params': grid_search.best_params_,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracy': accuracy
    }
    
    if r2 > best_score:
        best_score = r2
        best_model = best_pipeline
        best_model_name = name

# (No training output is printed)

# Save the best model and label encoders silently
joblib.dump(best_model, os.path.join(output_dir, "financial_wellbeing_model.pkl"))

joblib.dump(label_encoders, os.path.join(output_dir, "label_encoders.pkl"))

# ========================== PART 2: Customer Risk Assessment and Recommendations ========================== #

def get_score_category(score):
    if score >= 85:
        return "Excellent", "Your financial health is on a very solid foundation."
    elif score >= 70:
        return "Very Good", "Your financial health is strong, with some opportunities for improvement."
    elif score >= 60:
        return "Good", "Your financial health is above average, but there are several areas to strengthen."
    elif score >= 50:
        return "Fair", "Your financial health is moderate, with significant room for improvement."
    elif score >= 40:
        return "Needs Attention", "Your financial health requires immediate attention in several areas."
    else:
        return "Critical", "Your financial situation needs urgent intervention and comprehensive planning."

def generate_recommendations(customer_data):
    """
    Generate detailed personalized recommendations based on customer financial data.
    Uses Debt-to-Income Ratio in the checks.
    """
    income_stability = customer_data["Income Stability Score"]
    dti = customer_data["Debt-to-Income Ratio"]
    credit_utilization = customer_data["Credit Utilization Rate"]
    monthly_savings = customer_data["Monthly Savings Percentage"]
    financial_volatility = customer_data["Financial Volatility Index"]
    financial_resilience = customer_data["Financial Resilience Score"]
    financial_stress = customer_data["Financial Stress Index"]
    
    recommendations = {
        "Priority Actions": [],
        "Short-term Strategies": [],
        "Medium-term Goals": [],
        "Long-term Planning": []
    }
    
    if financial_stress > 2.5:
        recommendations["Priority Actions"].append("High financial stress detected. Consider debt consolidation or credit counseling.")
    if monthly_savings < 0.05:
        recommendations["Priority Actions"].append("Critical savings rate. Implement immediate expense reduction plan to free up at least 5% of income.")
    if financial_resilience < 1:
        recommendations["Priority Actions"].append("Emergency fund critically low. Prioritize building savings to cover at least 1 month of expenses.")
    if dti > 0.45:
        recommendations["Priority Actions"].append("Debt-to-income ratio is dangerously high. Consider debt management plan or debt consolidation.")
    
    if monthly_savings < 0.2:
        recommendations["Short-term Strategies"].append(
            f"Current savings rate ({monthly_savings*100:.1f}%) is below the recommended 20%. Review discretionary spending and implement a budgeting system.")
    if credit_utilization > 0.3:
        recommendations["Short-term Strategies"].append(
            f"Credit utilization ({credit_utilization*100:.1f}%) exceeds the recommended 30%. Create a plan to reduce balances on revolving accounts.")
    if financial_volatility > 0.15:
        recommendations["Short-term Strategies"].append("Cash flow volatility is high. Implement expense tracking and create more predictable payment schedules.")
    
    if financial_resilience < 3:
        recommendations["Medium-term Goals"].append(
            f"Current emergency fund covers {financial_resilience:.1f} months. Build it to cover 3-6 months of expenses.")
    if income_stability < 0.6:
        recommendations["Medium-term Goals"].append("Income stability needs improvement. Consider skills development or diversifying income sources.")
    if 0.2 <= monthly_savings < 0.3:
        recommendations["Medium-term Goals"].append("Increase savings rate to 25-30% through gradual expense reduction and income growth.")
    
    if financial_resilience >= 3 and monthly_savings >= 0.2:
        recommendations["Long-term Planning"].append("Consider diversifying investments based on your risk appetite and time horizon.")
    if dti < 0.3 and credit_utilization < 0.3:
        recommendations["Long-term Planning"].append("You're in a good position for major financial goals like home ownership or education funding.")
    if financial_stress < 1 and financial_resilience > 6:
        recommendations["Long-term Planning"].append("Consider opportunities for wealth building, such as real estate investment or business ventures.")
    
    for section in recommendations:
        if not recommendations[section]:
            if section == "Priority Actions":
                recommendations[section].append("No critical issues identified. Focus on optimization strategies.")
            elif section == "Short-term Strategies":
                recommendations[section].append("Maintain current financial habits while looking for optimization opportunities.")
            elif section == "Medium-term Goals":
                recommendations[section].append("Continue building financial resilience and consider increasing investment contributions.")
            elif section == "Long-term Planning":
                recommendations[section].append("Maintain current trajectory and regularly review long-term financial goals and investment strategy.")
    
    return recommendations

# ========================== PART 3: Custom Input for a Single Customer ========================== #

# Define a single customer's manual input (without "Risk Appetite")
customer_input = {
    "Monthly Income": 50000,
    "Savings & Investments": 32500,
    "Fixed Expenses": 7500,
    "Variable Expenses": 5000,
    "Existing Loans & Liabilities": 7500,
    # If DTI is not provided, it will be calculated
    "Debt-to-Income Ratio": None,
    "Monthly Cash Flow Trends": 2000,
    "Spending Behavior": "Balanced",
    "Investment Strategy": "Moderate"
}

def process_customer_input(customer_input):
    """
    Process raw customer input into model features.
    Calculates Debt-to-Income Ratio if not provided.
    """
    income = customer_input["Monthly Income"]
    savings = customer_input["Savings & Investments"]
    fixed_exp = customer_input["Fixed Expenses"]
    var_exp = customer_input["Variable Expenses"]
    loans = customer_input["Existing Loans & Liabilities"]
    # Calculate DTI if not provided
    dti = customer_input.get("Debt-to-Income Ratio")
    if dti is None:
        dti = loans / income
    cash_flow = customer_input["Monthly Cash Flow Trends"]
    
    income_stability = min(savings / income, 1.0)
    credit_utilization = min(loans / income, 1.0)
    monthly_savings_pct = max((income - (fixed_exp + var_exp)) / income, 0)
    financial_volatility = min(abs(cash_flow) / income, 1.0)
    financial_resilience = min(savings / (fixed_exp + var_exp + 1), 12)
    financial_stress = min(loans / (savings + 1), 5)
    
    customer_features = {
        "Income Stability Score": income_stability,
        "Debt-to-Income Ratio": dti,
        "Credit Utilization Rate": credit_utilization,
        "Monthly Savings Percentage": monthly_savings_pct,
        "Financial Volatility Index": financial_volatility,
        "Financial Resilience Score": financial_resilience,
        "Financial Stress Index": financial_stress,
        "Spending Behavior": customer_input["Spending Behavior"],
        "Investment Strategy": customer_input["Investment Strategy"]
    }
    
    return customer_features

customer_features = process_customer_input(customer_input)
input_df = pd.DataFrame({k: [v] for k, v in customer_features.items()})
for col in categorical_columns:
    if col in input_df.columns:
        # Safely transform categorical data
        input_df[col] = safe_label_transform(input_df[col], label_encoders[col])

predicted_score = best_model.predict(input_df)[0]
score_percentage = int(round(predicted_score * 100))
score_category, category_desc = get_score_category(score_percentage)
recommendations = generate_recommendations(customer_features)

# ========================== PART 4: Financial Health Report Generation ========================== #

def generate_financial_report(customer_input, customer_features, score_percentage, score_category, category_desc, recommendations):
    """Generate a comprehensive financial health report."""
    report = {
        "Customer Profile": {
            "Monthly Income": f"Rs.{customer_input['Monthly Income']:,.2f}",
            "Fixed Expenses": f"Rs.{customer_input['Fixed Expenses']:,.2f}",
            "Variable Expenses": f"Rs.{customer_input['Variable Expenses']:,.2f}",
            "Savings & Investments": f"Rs.{customer_input['Savings & Investments']:,.2f}",
            "Existing Loans & Liabilities": f"Rs.{customer_input['Existing Loans & Liabilities']:,.2f}",
            "Debt-to-Income Ratio": f"{customer_features['Debt-to-Income Ratio']*100:.1f}%",
            "Investment Approach": customer_input["Investment Strategy"],
            "Spending Behavior": customer_input["Spending Behavior"]
        },
        "Financial Health Score": {
            "Overall Score": score_percentage,
            "Category": score_category,
            "Description": category_desc
        },
        "Key Financial Metrics": {
            "Income Stability": f"{customer_features['Income Stability Score']*100:.1f}%",
            "Debt-to-Income Ratio": f"{customer_features['Debt-to-Income Ratio']*100:.1f}%",
            "Credit Utilization": f"{customer_features['Credit Utilization Rate']*100:.1f}%",
            "Monthly Savings Rate": f"{customer_features['Monthly Savings Percentage']*100:.1f}%",
            "Financial Resilience": f"{customer_features['Financial Resilience Score']:.1f} months",
            "Financial Stress Index": f"{customer_features['Financial Stress Index']:.2f} (0-5 scale)"
        },
        "Recommendations": recommendations,
        "Report Date": datetime.now().strftime("%Y-%m-%d")
    }
    return report

financial_report = generate_financial_report(customer_input, customer_features, score_percentage, score_category, category_desc, recommendations)

def print_financial_report(report):
    """Print the financial report in a structured format."""
    print("\n" + "="*80)
    print("FINANCIAL WELL-BEING REPORT".center(80))
    print("="*80)
    
    print("\nCUSTOMER PROFILE:")
    for key, value in report["Customer Profile"].items():
        print(f"  {key}: {value}")
    
    print("\nFINANCIAL HEALTH ASSESSMENT:")
    print(f"  Overall Score: {report['Financial Health Score']['Overall Score']} / 100 ({report['Financial Health Score']['Category']})")
    print(f"  {report['Financial Health Score']['Description']}")
    
    print("\nKEY FINANCIAL METRICS:")
    for key, value in report["Key Financial Metrics"].items():
        print(f"  {key}: {value}")
    
    print("\nRECOMMENDATIONS:")
    for section, items in report["Recommendations"].items():
        print(f"\n  {section}:")
        for i, item in enumerate(items, 1):
            print(f"    {i}. {item}")
    
    print("\nFINANCIAL OPPORTUNITIES:")
    if report["Financial Health Score"]["Overall Score"] >= 70:
        print("  - Consider increasing investment contributions for long-term wealth building")
        print("  - Explore tax optimization strategies for your investments")
        print("  - Evaluate opportunities for further income diversification")
    elif report["Financial Health Score"]["Overall Score"] >= 50:
        print("  - Focus on debt reduction to improve financial flexibility")
        print("  - Build emergency fund to at least 6 months of expenses")
        print("  - Consider upskilling to increase earning potential")
    else:
        print("  - Prioritize building basic financial stability before major investment decisions")
        print("  - Consider credit counseling or financial literacy resources")
        print("  - Focus on increasing income and reducing high-interest debt")
    
    print("\nDISCLAIMER:")
    print("  This report is for informational purposes only and does not constitute financial advice.")
    print("  Please consult with a qualified financial advisor for personalized guidance.")
    
    print("\n" + "="*80)
    print(f"Report generated on: {report['Report Date']}".center(80))
    print("="*80 + "\n")

print_financial_report(financial_report)
with open(os.path.join(output_dir, "financial_health_report.txt"), 'w', encoding='utf-8') as f:
    import sys
    original_stdout = sys.stdout
    sys.stdout = f
    print_financial_report(financial_report)
    sys.stdout = original_stdout

print(f"\nFinancial health report saved to {os.path.join(output_dir, 'financial_health_report.txt')}")

# ========================== PART 5: Financial Simulation & Scenario Analysis ========================== #

def simulate_financial_scenarios(customer_input, best_model, label_encoders):
    """Simulate different financial scenarios and their impact on the well-being score."""
    base_features = process_customer_input(customer_input)
    base_df = pd.DataFrame({k: [v] for k, v in base_features.items()})
    
    for col in categorical_columns:
        if col in base_df.columns:
            base_df[col] = safe_label_transform(base_df[col], label_encoders[col])
    
    base_score = best_model.predict(base_df)[0] * 100
    
    scenarios = {
        "Increase Savings Rate by 10%": {
            "Monthly Savings Percentage": min(base_features["Monthly Savings Percentage"] + 0.1, 1.0)
        },
        "Reduce Debt-to-Income Ratio by 5%": {
            "Debt-to-Income Ratio": max(base_features["Debt-to-Income Ratio"] - 0.05, 0.0)
        },
        "Reduce Credit Utilization by 10%": {
            "Credit Utilization Rate": max(base_features["Credit Utilization Rate"] - 0.1, 0.0)
        },
        "Build Emergency Fund (Double Resilience)": {
            "Financial Resilience Score": min(base_features["Financial Resilience Score"] * 2, 12),
            "Income Stability Score": min(base_features["Income Stability Score"] + 0.1, 1.0)
        },
        "Increase Income by 20%": {
            "Monthly Savings Percentage": min(base_features["Monthly Savings Percentage"] + 0.1, 1.0),
            "Financial Resilience Score": min(base_features["Financial Resilience Score"] * 1.2, 12),
            "Credit Utilization Rate": max(base_features["Credit Utilization Rate"] * 0.8, 0.0),
            "Debt-to-Income Ratio": max(base_features["Debt-to-Income Ratio"] * 0.8, 0.0)
        }
    }
    
    scenario_results = []
    
    for scenario_name, changes in scenarios.items():
        scenario_features = base_features.copy()
        for feature, value in changes.items():
            scenario_features[feature] = value
        
        scenario_df = pd.DataFrame({k: [v] for k, v in scenario_features.items()})
        for col in categorical_columns:
            if col in scenario_df.columns:
                scenario_df[col] = safe_label_transform(scenario_df[col], label_encoders[col])
        
        scenario_score = best_model.predict(scenario_df)[0] * 100
        score_change = scenario_score - base_score
        
        scenario_results.append({
            "Scenario": scenario_name,
            "Score": scenario_score,
            "Change": score_change,
            "Percent Change": (score_change / base_score) * 100
        })
    
    scenarios_df = pd.DataFrame(scenario_results).sort_values(by="Change", ascending=False)
    return scenarios_df, base_score

scenario_results, base_score = simulate_financial_scenarios(customer_input, best_model, label_encoders)

print("\n" + "="*80)
print("FINANCIAL SCENARIO ANALYSIS".center(80))
print("="*80)
print(f"\nBase Financial Well-being Score: {base_score:.1f}")
print("\nPotential Improvement Scenarios:")

for _, row in scenario_results.iterrows():
    print(f"  {row['Scenario']}:")
    print(f"    New Score: {row['Score']:.1f} ({row['Change']:+.1f} points, {row['Percent Change']:+.1f}%)")

print("\nAnalysis complete. All outputs saved to", output_dir)
