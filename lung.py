import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

#--------------------loading the data------------------------------------------
lung_df = pd.read_csv("survey lung cancer.csv")

#---------------------print properties----------------------------------------
print(lung_df.head())
print(lung_df.shape)

#-----------------------------drop the duplicates-------------------------------
lung_df = lung_df.drop_duplicates()
print(f"Duplicates remaining: {lung_df.duplicated().sum()}")

#-------------------labelencoder (FIXED: Isolated encoders per column to prevent overwriting)-------------------
category_cols = ['GENDER', 'LUNG_CANCER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 
                 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

encoders = {}
for col in category_cols:
    le = LabelEncoder()
    lung_df[col] = le.fit_transform(lung_df[col])
    encoders[col] = le  # Saved for the predictive system and web app mapping

print(lung_df['LUNG_CANCER'].value_counts())

#-------------------feature engineering interaction element-------------------
lung_df['ANXYELFIN'] = lung_df['ANXIETY'] * lung_df['YELLOW_FINGERS']

#-------------------correlation analysis---------------------------------
cm = lung_df.corr()
sns.heatmap(cm, annot=True, cbar=True, fmt='.2f', cmap='Set2')
plt.show()

#--------------------------split the x and y-------------------
X = lung_df.drop(columns=['LUNG_CANCER'])
Y = lung_df['LUNG_CANCER']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#--------------------------Standard Scaling----------------------------
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#-----------------------------building the model---------------------------------------
model = RandomForestClassifier(random_state=42,
                               n_estimators=40,
                               max_depth=5,
                               min_samples_split=12,
                               min_samples_leaf=5,
                               criterion='gini')

#-------------------------fit the model to train it------------------------------
model.fit(X_train_scaler, Y_train)

#-------------------------testing_predictions---------------------------------------------
Y_testing_prediction = model.predict(X_test_scaler)
print(f"XGBoost Test Accuracy: {accuracy_score(Y_test, Y_testing_prediction)*100:.2f}%")
print(classification_report(Y_test, Y_testing_prediction))

#---------------------------------------EXPORT PIPELINE STRUCTURES-----------------
lung_assets = {
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'feature_columns': list(X.columns)
}

joblib.dump(lung_assets, "lung_assets.pkl")
print("All lung cancer XGBoost pipeline assets successfully generated and exported to 'lung_assets.pkl'!")
print('='*60)

# --------------------------------------------------------------------------------------------------
# 🧪 BUILDING A PREDICTIVE SYSTEM TO TEST THE MODEL (ADDED)
# --------------------------------------------------------------------------------------------------
# 1. Define raw test patient string in the original dataset's exact row layout format
# Schema order: GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, 
#               FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, 
#               SWALLOWING DIFFICULTY, CHEST PAIN
raw_input_data = "M, 69, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2"

# 2. Parse the string into a clean list of text items, trimming trailing spaces
raw_features = [item.strip() for item in raw_input_data.split(',')]

# 3. Create a structured dictionary mapping values back to their specific features
input_dict = {
    'GENDER': encoders['GENDER'].transform([raw_features[0]])[0],
    'AGE': int(raw_features[1]),
    'SMOKING': encoders['SMOKING'].transform([raw_features[2]])[0],
    'YELLOW_FINGERS': encoders['YELLOW_FINGERS'].transform([raw_features[3]])[0],
    'ANXIETY': encoders['ANXIETY'].transform([raw_features[4]])[0],
    'PEER_PRESSURE': encoders['PEER_PRESSURE'].transform([raw_features[5]])[0],
    'CHRONIC DISEASE': encoders['CHRONIC DISEASE'].transform([raw_features[6]])[0],
    'FATIGUE ': encoders['FATIGUE '].transform([raw_features[7]])[0],
    'ALLERGY ': encoders['ALLERGY '].transform([raw_features[8]])[0],
    'WHEEZING': encoders['WHEEZING'].transform([raw_features[9]])[0],
    'ALCOHOL CONSUMING': encoders['ALCOHOL CONSUMING'].transform([raw_features[10]])[0],
    'COUGHING': encoders['COUGHING'].transform([raw_features[11]])[0],
    'SHORTNESS OF BREATH': encoders['SHORTNESS OF BREATH'].transform([raw_features[12]])[0],
    'SWALLOWING DIFFICULTY': encoders['SWALLOWING DIFFICULTY'].transform([raw_features[13]])[0],
    'CHEST PAIN': encoders['CHEST PAIN'].transform([raw_features[14]])[0]
}

# 4. Generate the engineered interaction column variable dynamically
input_dict['ANXYELFIN'] = input_dict['ANXIETY'] * input_dict['YELLOW_FINGERS']

# 5. Represent entry array as a DataFrame matching X's feature scheme column sequence
input_df = pd.DataFrame([input_dict])[list(X.columns)]

# 6. Apply standard scaling standardization step using your trained scaling parameters
input_scaled = scaler.transform(input_df)

# 7. Complete prediction inference
prediction = model.predict(input_scaled)
print(f"Predictive System Numerical Output: {prediction}")

# 8. Inverse transform numerical output back to original string target classification ('YES' or 'NO')
predicted_label = encoders['LUNG_CANCER'].inverse_transform(prediction)
print(f"Clinical Risk Classification Result: Patient Lung Cancer Screening is **{predicted_label[0]}**")
#---------------------------------------EXPORT PIPELINE STRUCTURES (ADDED)-----------------
lung_assets = {
    'model': model,
    'scaler': scaler,
    'encoders': encoders,
    'feature_columns': list(X.columns)
}

joblib.dump(lung_assets, "lungs_model.pkl")
print("All lung cancer XGBoost pipeline assets successfully generated and exported to 'lung_assets.pkl'!")