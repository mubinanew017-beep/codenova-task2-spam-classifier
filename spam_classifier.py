
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Set the dataset path
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "SMSSpamCollection"

# 2. Load the dataset
data = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))

print("\nLabel distribution:")
print(data["label"].value_counts())

# 3. Separate messages and labels
X = data["message"]
y = data["label"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# 5. Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)),
    ("classifier", MultinomialNB())
])

# 6. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training completed!")

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("\n--- MODEL EVALUATION ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_test, predictions)

print("\nClassification Report:")
print(report)

# 9. Create the confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
)

print("Confusion Matrix:")
print(cm)

# 10. Save the evaluation report
results_path = BASE_DIR / "evaluation_report.txt"

with open(results_path, "w", encoding="utf-8") as file:
    file.write("SMS SPAM CLASSIFIER - EVALUATION REPORT\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Total dataset messages: {len(data)}\n")
    file.write(f"Training messages: {len(X_train)}\n")
    file.write(f"Testing messages: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

print("\nEvaluation report saved:", results_path.name)

# 11. Create and save the confusion matrix image
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("SMS Spam Classifier - Confusion Matrix")
plt.tight_layout()

image_path = BASE_DIR / "confusion_matrix.png"
plt.savefig(image_path, dpi=300)
plt.close()

print("Confusion matrix image saved:", image_path.name)

# 12. Test custom messages
print("\n--- TEST YOUR OWN MESSAGES ---")
print("Enter 'exit' to stop.")

while True:
    message = input("\nEnter an SMS message: ").strip()

    if message.lower() == "exit":
        print("Classifier closed. Thank you!")
        break

    if not message:
        print("Please enter a message.")
        continue

    prediction = model.predict([message])[0]

    if prediction == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")
