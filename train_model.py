import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def train_model():

    # Load small dataset
    data = pd.read_csv("adult_small.csv")

    # Clean columns
    data.columns = data.columns.str.strip()

    # Remove missing values
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Only use features used in the app
    features = ["age", "education", "occupation", "hours-per-week"]

    X = data[features]
    y = data["income"]

    categorical_cols = ["education", "occupation"]

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
