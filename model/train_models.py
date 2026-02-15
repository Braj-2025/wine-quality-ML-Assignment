# Let's train teh dataset for all 6 models to calculate metrics as asked


from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    results.append([
        name,
        accuracy_score(y_test, preds),
        roc_auc_score(y_test, probs),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds),
        matthews_corrcoef(y_test, preds)
    ])

    joblib.dump(model, f"{name}.pkl")

columns = ["ML Model Name","Accuracy","AUC","Precision","Recall","F1","MCC"]
results_df = pd.DataFrame(results, columns=columns)

results_df.to_csv("model_comparison.csv", index=False)
results_df
