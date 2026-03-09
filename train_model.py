import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("adult.csv")

# Fix column names
data.columns = data.columns.str.replace('.', '-', regex=False)

# Remove missing values
data.replace(" ?", pd.NA, inplace=True)
data.dropna(inplace=True)

# Features and target
X = data.drop("income", axis=1)
y = data["income"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include="object").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "salary_pipeline.pkl")

print("Model trained and saved successfully!")