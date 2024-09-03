import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    label_encoders = {}
    for column in ['Gender', 'Disability Type', 'Disability Severity', 'Fitness Level', 
                   'Preferred Exercise Type', 'Health Conditions', 'Goals']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le
    return data, label_encoders

def feature_engineering(data):
    X = data.drop(columns=['UserID', 'Preferred Exercise Type'])
    y = data['Preferred Exercise Type']
    X['Age_Disability_Interaction'] = X['Age'] * X['Disability Severity']
    return X, y

def resample_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(X_resampled, y_resampled):
    gb_clf = GradientBoostingClassifier(random_state=42)
    svc_clf = SVC(probability=True, random_state=42)
    ensemble_clf = VotingClassifier(estimators=[('gb', gb_clf), ('svc', svc_clf)], voting='soft')

    param_grid = {
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf']
    }

    grid_search = GridSearchCV(ensemble_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    return grid_search.best_estimator_

def evaluate_model(best_model, X_resampled, y_resampled):
    cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    final_accuracy = np.mean(cv_scores)
    print(f'Cross-validated accuracy: {final_accuracy:.2f}')

def test_model(best_model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_pred = best_model.predict(X_test)
    final_classification_report = classification_report(y_test, y_test_pred)
    print(final_classification_report)
    return X_train, X_test, y_train, y_test

def save_model(model, filename):
    joblib.dump(model, filename)

def preprocess_new_user_data(new_user_data, label_encoders, X_columns):
    new_user_data_encoded = {}
    for column, le in label_encoders.items():
        if column in new_user_data:
            try:
                new_user_data_encoded[column] = le.transform([new_user_data[column]])[0]
            except ValueError:
                print(f"Warning: '{new_user_data[column]}' is not a known label for column '{column}'")
                new_user_data_encoded[column] = le.transform([le.classes_[0]])[0]  # Default value

    new_user_df = pd.DataFrame([new_user_data_encoded])
    for col in X_columns:
        if col not in new_user_df.columns:
            new_user_df[col] = 0  # Default value for missing columns

    if 'Age' in new_user_df.columns and 'Disability Severity' in new_user_df.columns:
        new_user_df['Age_Disability_Interaction'] = new_user_df['Age'] * new_user_df['Disability Severity']
    else:
        new_user_df['Age_Disability_Interaction'] = 0  # Default value if columns are missing

    new_user_df = new_user_df[X_columns]
    return new_user_df

def generate_workout_plan(predicted_exercise_type_label):
    workout_plans = {
        'Cardio': '30 minutes of cycling, 20 minutes of swimming, and 10 minutes of stretching.',
        'Strength': '3 sets of 10 reps of seated leg press, 3 sets of 12 reps of lat pulldowns, and 3 sets of 15 reps of chest presses.',
        'Flexibility': '30 minutes of yoga, 15 minutes of dynamic stretching, and 10 minutes of static stretching.',
        'Balance': '15 minutes of standing leg raises, 15 minutes of stability ball exercises, and 10 minutes of Tai Chi.',
        'Endurance': '40 minutes of brisk walking, 20 minutes of rowing machine, and 10 minutes of cool-down exercises.'
    }
    return workout_plans.get(predicted_exercise_type_label, "Custom workout plan based on specific needs.")

def main():
    # Load and preprocess data
    data, label_encoders = preprocess_data("AI_Fitness_App_PWD_Dataset_Large.xlsx")
    
    # Feature engineering
    X, y = feature_engineering(data)
    
    # Resample data
    X_resampled, y_resampled = resample_data(X, y)
    
    # Train model
    best_model = train_model(X_resampled, y_resampled)
    
    # Evaluate model
    evaluate_model(best_model, X_resampled, y_resampled)
    
    # Test model
    X_train, X_test, y_train, y_test = test_model(best_model, X, y)
    
    # Save model
    save_model(best_model, 'personalized_workout_model.pkl')
    
    # Example new user profile
    new_user_data = {
        'Gender': 'Male',
        'Age': 18,
        'Disability Type': 'Mobility Impairment',
        'Disability Severity': 'Moderate',
        'Fitness Level': 'Beginner',
        'Health Conditions': 'None',
        'Goals': 'Pain Management'
    }
    
    # Preprocess new user data
    new_user_df = preprocess_new_user_data(new_user_data, label_encoders, X.columns)
    
    # Predict exercise type
    predicted_exercise_type = best_model.predict(new_user_df)[0]
    predicted_exercise_type_label = label_encoders['Preferred Exercise Type'].inverse_transform([predicted_exercise_type])[0]
    
    # Generate and print workout plan
    workout_plan = generate_workout_plan(predicted_exercise_type_label)
    print(f"Predicted Exercise Type: {predicted_exercise_type_label}")
    print(f"Generated Workout Plan: {workout_plan}")

if __name__ == "__main__":
    main()
