import argparse
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    # Loads the CSV. Assumes the file has headers (age, sex, cp, etc.) on row 1.
    # If your CSV has text descriptions at the top, delete them in Excel first!
    df = pd.read_csv(path)
    return df

def run_training(df: pd.DataFrame, show_plots: bool = True):
    print("Training Model...")
    print("Missing values:\n", df.isnull().sum())
    df = df.drop_duplicates()

    if show_plots:
        # Note: The script pauses here until you close the plot windows
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.show()

        sns.countplot(x='target', data=df)
        plt.title('Distribution of Heart Disease Presence (0=No, 1=Yes)')
        plt.show()

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, scaler, X.columns.tolist()

def predict_heart_disease(model, scaler, feature_names: list, **features):
    """Make a prediction for a patient with given features."""
    feature_dict = {name: features.get(name, 0) for name in feature_names}
    X_sample = pd.DataFrame([feature_dict])
    X_scaled = scaler.transform(X_sample)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    result = "Heart Disease Risk" if prediction == 1 else "No Heart Disease Risk"
    print(f"\nPrediction: {result}")
    print(f"Confidence (No Disease): {probability[0]:.2%}")
    print(f"Confidence (Disease): {probability[1]:.2%}")

def interactive_predict(model, scaler, feature_names: list):
    """Interactive mode to predict for multiple patients."""
    print("\n" + "="*60)
    print("Interactive Prediction Mode")
    print("="*60)
    print(f"Features to provide: {', '.join(feature_names)}")
    print("Type 'quit' to exit\n")
    
    while True:
        print("\nEnter patient data (comma-separated values in order):")
        print(f"Example: 58, 1, 0, 100, 234, 0, 1, 156, 0, 0.1, 2, 1, 3")
        user_input = input("Data: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != len(feature_names):
                print(f"Error: Expected {len(feature_names)} values, got {len(values)}")
                continue
            
            features = {name: val for name, val in zip(feature_names, values)}
            predict_heart_disease(model, scaler, feature_names, **features)
        except ValueError:
            print("Error: Please enter valid numbers separated by commas")

def parse_args():
    p = argparse.ArgumentParser(description='Train RandomForest on heart disease dataset')
    p.add_argument('--data', '-d', default='heart.csv', help='Path to the CSV dataset')
    p.add_argument('--no-plots', action='store_true', help='Disable plotting (useful for headless runs)')
    p.add_argument('--predict', nargs='+', help='Predict with features: --predict age=22 sex=1 cp=0 ...')
    p.add_argument('--interactive', '-i', action='store_true', help='Interactive mode for multiple predictions')
    return p.parse_args()

def main():
    args = parse_args()
    try:
        df = load_dataset(args.data)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print('Provide dataset using --data PATH or place heart.csv next to this script.', file=sys.stderr)
        sys.exit(2)

    model, scaler, feature_names = run_training(df, show_plots=not args.no_plots)
    
    # --- LOGIC UPDATE ---
    # This logic forces the script to wait for your input automatically!
    if args.predict:
        # If you supplied arguments in command line (rare)
        features = {}
        for item in args.predict:
            key, val = item.split('=')
            features[key] = float(val)
        predict_heart_disease(model, scaler, feature_names, **features)
    else:
        # DEFAULT: Open the interactive question mode
        interactive_predict(model, scaler, feature_names)

if __name__ == '__main__':
    main()