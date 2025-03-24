import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_excel('wdbc.xls', header=None, skiprows=1)

if df.shape[1] == 1:
    df = df[0].str.split(',', expand=True)

df['Class'] = df[1].map({'M': 1, 'B': 0})  
X = df.iloc[:, 2:].astype(float)          
X.columns = X.columns.astype(str)

y = df['Class']

clf = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42
)
clf.fit(X, y)

tree = clf.tree_
root_node = 0
split_feature_idx = tree.feature[root_node]
split_threshold = tree.threshold[root_node]

def calculate_metrics(y_true):
    p = np.mean(y_true)  
    if p == 0 or p == 1:
        entropy = 0
    else:
        entropy = -p * np.log2(p) - (1-p)*np.log2(1-p)
    gini = 1 - (p**2 + (1-p)**2)
    error = min(p, 1-p)
    return entropy, gini, error

root_entropy, root_gini, root_error = calculate_metrics(y)

left_mask = X.iloc[:, split_feature_idx] <= split_threshold
right_mask = ~left_mask

left_entropy, left_gini, left_error = calculate_metrics(y[left_mask])
right_entropy, right_gini, right_error = calculate_metrics(y[right_mask])

n_left = sum(left_mask)
n_right = sum(right_mask)
n_total = len(y)

weighted_child_entropy = (n_left/n_total)*left_entropy + (n_right/n_total)*right_entropy
information_gain = root_entropy - weighted_child_entropy

print("\nFirst Split Point Analysis：")
print(f"Splitting Feature Index: {split_feature_idx}")
print(f"Splitting Threshold: {split_threshold:.4f}")
print(f"\nRoot Node Metrics:")
print(f"Entropy: {root_entropy:.4f}")
print(f"Gini Index: {root_gini:.4f}")
print(f"Misclassification Error: {root_error:.4f}")
print(f"\nInformation Gain After Split: {information_gain:.4f}")