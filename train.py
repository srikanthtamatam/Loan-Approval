import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("C:\\Users\\DELL\\Desktop\\8th_sem\\loan_approval_dataset.csv")

# Clean column names and drop 'loan_id'
df.columns = df.columns.str.strip()
df = df.drop(columns=['loan_id'])

# Separate features and target
X = df.drop(columns=['loan_status'])
y = df['loan_status'].str.strip()

# Encode categorical features
X['education'] = LabelEncoder().fit_transform(X['education'].str.strip())
X['self_employed'] = LabelEncoder().fit_transform(X['self_employed'].str.strip())


# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Identify best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Save the best model and scaler
with open("C:\\Users\\DELL\\Desktop\\best_model11.pkl", "wb") as f:
    pickle.dump((best_model, scaler), f)

# Save model accuracies plot with values
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)

# Add accuracy values on top of bars
for i, v in enumerate(accuracies.values()):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("C:\\Users\\DELL\\Desktop\\model_accuracies11.png")
plt.close()

# Save feature importance plot if supported
if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=best_model.feature_importances_, y=X.columns, palette="crest")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.tight_layout()
    plt.savefig("C:\\Users\\DELL\\Desktop\\feature_importance11.png")
    plt.close()
