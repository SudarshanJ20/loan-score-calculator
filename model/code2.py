import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

# ------------------ Core Functions ------------------ #
def get_user_input():
    """Safely collect user financial data with validation"""
    print("\n" + "="*50)
    print("FINANCIAL HEALTH ASSESSMENT".center(50))
    print("="*50)
    
    while True:
        try:
            data = {
                'Monthly Income': float(input("\nMonthly Income (₹): ")),
                'Savings & Investments': float(input("Savings & Investments (₹): ")),
                'Fixed Expenses': float(input("Fixed Expenses (₹): ")),
                'Variable Expenses': float(input("Variable Expenses (₹): ")),
                'Existing Loans & Liabilities': float(input("Loans/Liabilities (₹): ")),
                'Spending Behavior': input("Spending (Conservative/Balanced/Aggressive): ").title(),
                'Investment Strategy': input("Investments (Conservative/Moderate/Aggressive): ").title()
            }
            
            # Validate inputs
            if any(val < 0 for val in [data['Monthly Income'], data['Savings & Investments'], 
                                      data['Fixed Expenses'], data['Variable Expenses'], 
                                      data['Existing Loans & Liabilities']]):
                raise ValueError("Negative values not allowed")
                
            if data['Spending Behavior'] not in ['Conservative', 'Balanced', 'Aggressive']:
                raise ValueError("Invalid spending behavior")
                
            if data['Investment Strategy'] not in ['Conservative', 'Moderate', 'Aggressive']:
                raise ValueError("Invalid investment strategy")
                
            return data
            
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def calculate_metrics(data):
    """Calculate financial ratios"""
    income = max(data['Monthly Income'], 0.01)  # Prevent division by zero
    return {
        'Income Stability': min(data['Savings & Investments'] / income, 1.0),
        'Debt-to-Income': min(data['Existing Loans & Liabilities'] / income, 1.0),
        'Savings Rate': max((income - (data['Fixed Expenses'] + data['Variable Expenses'])) / income, 0)
    }

def initialize_model():
    """Initialize with fallback model if no training data exists"""
    # Try to load pre-trained model
    if os.path.exists("model/financial_model.pkl"):
        return joblib.load("model/financial_model.pkl"), joblib.load("model/label_encoders.pkl")
    
    # Fallback model
    print("\nNo training data found. Using built-in assessment model")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Minimal training data
    X = pd.DataFrame({
        'Income Stability': [0.3, 0.6, 0.9],
        'Debt-to-Income': [0.5, 0.3, 0.1],
        'Savings Rate': [0.1, 0.2, 0.3],
        'Spending': [0, 1, 1],  # Conservative, Balanced, Balanced
        'Investing': [1, 1, 2]  # Moderate, Moderate, Aggressive
    })
    y = np.array([0.4, 0.7, 0.9])  # Sample scores
    
    model.fit(X, y)
    
    # Label encoders
    label_encoders = {
        'Spending': LabelEncoder().fit(['Conservative', 'Balanced', 'Aggressive']),
        'Investing': LabelEncoder().fit(['Conservative', 'Moderate', 'Aggressive'])
    }
    
    return model, label_encoders

def generate_report(user_data, metrics, score):
    """Create comprehensive financial report"""
    report = f"""
    FINANCIAL HEALTH REPORT
    {'='*50}
    INCOME & EXPENSES:
    - Monthly Income: ₹{user_data['Monthly Income']:,.2f}
    - Fixed Expenses: ₹{user_data['Fixed Expenses']:,.2f}
    - Variable Expenses: ₹{user_data['Variable Expenses']:,.2f}
    
    ASSETS & LIABILITIES:
    - Savings: ₹{user_data['Savings & Investments']:,.2f}
    - Debt: ₹{user_data['Existing Loans & Liabilities']:,.2f}
    
    KEY METRICS:
    - Income Stability: {metrics['Income Stability']*100:.1f}%
    - Debt-to-Income: {metrics['Debt-to-Income']*100:.1f}%
    - Savings Rate: {metrics['Savings Rate']*100:.1f}%
    
    FINANCIAL SCORE: {score}/100
    {'='*50}
    RECOMMENDATIONS:
    """
    
    # Generate recommendations
    if metrics['Savings Rate'] < 0.1:
        report += "1. 🚨 CRITICAL: Increase savings immediately (currently under 10%)\n"
    elif metrics['Savings Rate'] < 0.2:
        report += "1. ⚠️ Warning: Boost savings to at least 20%\n"
    else:
        report += "1. ✅ Good savings rate - maintain this\n"
    
    if metrics['Debt-to-Income'] > 0.4:
        report += "2. 🚨 CRITICAL: Reduce debt (ratio over 40%)\n"
    elif metrics['Debt-to-Income'] > 0.3:
        report += "2. ⚠️ Warning: Lower debt ratio below 30%\n"
    
    if user_data['Spending Behavior'] == 'Aggressive':
        report += "3. 💡 Tip: Track discretionary spending\n"
    
    report += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    report += "\n" + "="*50
    
    return report

# ------------------ Main Program ------------------ #
def main():
    # Setup
    os.makedirs("reports", exist_ok=True)
    model, encoders = initialize_model()
    
    while True:
        print("\nOptions:")
        print("1. Analyze my finances")
        print("2. Exit")
        choice = input("Select (1/2): ").strip()
        
        if choice == '1':
            try:
                # Get and process user data
                user_data = get_user_input()
                metrics = calculate_metrics(user_data)
                
                # Prepare for prediction
                input_data = pd.DataFrame([{
                    'Income Stability': metrics['Income Stability'],
                    'Debt-to-Income': metrics['Debt-to-Income'],
                    'Savings Rate': metrics['Savings Rate'],
                    'Spending': encoders['Spending'].transform([user_data['Spending Behavior']])[0],
                    'Investing': encoders['Investing'].transform([user_data['Investment Strategy']])[0]
                }])
                
                # Predict score (0-100 scale)
                score = int(round(model.predict(input_data)[0] * 100))
                
                # Generate and save report
                report = generate_report(user_data, metrics, score)
                filename = f"reports/financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(filename, 'w') as f:
                    f.write(report)
                
                print("\n" + "="*50)
                print(report)
                print(f"\nReport saved to {filename}")
                print("="*50)
                
            except Exception as e:
                print(f"\nError: {str(e)}\nPlease try again")
                
        elif choice == '2':
            print("\nThank you for using the Financial Health Analyzer!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1 or 2")

if __name__ == "__main__":
    main()