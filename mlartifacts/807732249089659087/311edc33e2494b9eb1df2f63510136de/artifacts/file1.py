import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# set the tracking uri to local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 11

# set the experiment name
mlflow.set_experiment("Wine_Quality_Experiment")

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, random_state=42)
    rf.fit(X=X_train, y=y_train)

    y_pred = rf.predict(X=X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # creating the confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # save the plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__) # can even log the current script

    # log tags
    mlflow.set_tags({"Author":"Manjesh","Project":"Wine classification"}) # can log the tag as dictionary, which can used as top level identification
    # of experiments and runs
    
    # log model
    mlflow.sklearn.load_model(rf, "Random-Forest-Model")

    print(mlflow.get_tracking_uri()) # http://127.0.0.1:5000