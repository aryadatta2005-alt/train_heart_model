import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt  # Added for graphs
import seaborn as sns            # Added for graphs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. LOAD AND VISUALIZE DATA
# ==========================================
print("Loading data...")

try:
    df = pd.read_csv('heart.csv')
    
    # Clean duplicates and nulls
    df.drop_duplicates(inplace=True)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # === GRAPHS SECTION ===
    print("Generating graphs... (Close the windows to continue)")
    
    # Graph 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()  # <--- Script pauses here until you close the window
    
    # Graph 2: Target Count
    sns.countplot(x='target', data=df)
    plt.title('Heart Disease Count (0=No, 1=Yes)')
    plt.show()  # <--- Script pauses here again
    # ======================

    # Split Data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save Files
    print("Saving model and scaler...")
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Files saved successfully!")

except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please put the dataset in this folder.")
    exit()

# ==========================================
# 2. PREDICTION LOOP
# ==========================================

print("\n" + "="*60)
print(" INTERACTIVE PREDICTION MODE")
print("="*60)
print("Type 'exit' to quit.\n")

def get_prediction_result(input_data):
    input_as_numpy = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_as_numpy)
    
    prediction = model.predict(std_data)
    probability = model.predict_proba(std_data)
    
    if prediction[0] == 0:
        return f"🟢 Result: HEALTHY (Confidence: {probability[0][0]*100:.2f}%)"
    else:
        return f"🔴 Result: HEART DISEASE DETECTED (Confidence: {probability[0][1]*100:.2f}%)"

while True:
    print("\nEnter patient data (13 numbers separated by comma):")
    print("Example: 63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1")
    
    user_input = input("Data > ").strip()
    
    if user_input.lower() == 'exit':
        break
        
    try:
        input_data = [float(x.strip()) for x in user_input.split(',')]
        if len(input_data) != 13:
            print(f"⚠️ Error: You entered {len(input_data)} values. We need exactly 13.")
            continue
        result = get_prediction_result(input_data)
        print(result)
    except ValueError:
        print("⚠️ Error: Please enter numbers only.")
