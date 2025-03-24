from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate depths
results = []
for depth in range(1, 6):
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Calculate metrics (using macro average for consistency)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results.append({
        'Depth': depth,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

# Print results
import pandas as pd
df = pd.DataFrame(results)
print(df)