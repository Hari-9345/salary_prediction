import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_model():

    data = pd.read_csv("adult.csv")

    # Fix column names
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(".", "-", regex=False)

    # Clean missing values
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)

    X = data.drop("income", axis=1)
    y = data["income"]

    categorical_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    return pipeline
