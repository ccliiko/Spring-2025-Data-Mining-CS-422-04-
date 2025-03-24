import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def basic_decision_tree_analysis(filepath):
    df = pd.read_excel(filepath, header=None, skiprows=1)
    
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
    
    return X, y, clf

def pca_analysis(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    def evaluate_model(X_transformed):
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.3, random_state=42)
        
        clf = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=2,
            min_samples_split=5,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        FP, TP = cm[0,1], cm[1,1]
        return {
            'F1': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'FP': FP,
            'TP': TP,
            'FPR': FP/(FP+cm[0,0]),
            'TPR': TP/(TP+cm[1,0])
        }
    
    results = {}
    for n in [None, 1, 2]:
        if n is None:
            X_transformed = X_scaled
            name = "Original"
        else:
            pca = PCA(n_components=n)
            X_transformed = pca.fit_transform(X_scaled)
            name = f"PC{n}"
        
        results[name] = evaluate_model(X_transformed)
    
    return pd.DataFrame(results).T

if __name__ == "__main__":
    file_path = "D:/python3/project/wdbc.xls"
    X, y, clf = basic_decision_tree_analysis(file_path)
    results = pca_analysis(X, y)
    print(results)