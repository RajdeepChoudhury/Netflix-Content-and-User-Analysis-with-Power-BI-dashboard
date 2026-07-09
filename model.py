import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("netflix_titles.csv")

mask = df["rating"].astype(str).str.contains(r"\d+\s*min|\d+\s*Season", regex=True)
df.loc[mask, "rating"] = None

df["rating"] = df["rating"].fillna("Unknown")

df["duration_numeric"] = (
    df["duration"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

rating_counts = df["rating"].value_counts()
valid_ratings = rating_counts[rating_counts >= 2].index
df = df[df["rating"].isin(valid_ratings)]

features = [
    "type",
    "director",
    "cast",
    "country",
    "release_year",
    "duration_numeric",
    "listed_in"
]

X = df[features]
y = df["rating"]

categorical_features = [
    "type",
    "director",
    "cast",
    "country",
    "listed_in"
]

numerical_features = [
    "release_year",
    "duration_numeric"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            categorical_features
        ),
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            ),
            numerical_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print(f"Model Accuracy : {accuracy * 100:.2f}%")
print("=" * 60)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

for content in df["type"].unique():
    subset = df[df["type"] == content]

    plt.scatter(
        subset["release_year"],
        subset["duration_numeric"],
        alpha=0.5,
        s=18,
        label=content
    )

plt.title("Netflix Content Concentration")
plt.xlabel("Release Year")
plt.ylabel("Duration (Minutes / Seasons)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()