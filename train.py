from data import dataframe
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(uri="your_tracking_uri")

experiment_name = "your_experiment_name"

experiment = mlflow.get_experiment_by_name(name=experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(name=experiment_name)
else:
    experiment_id = experiment.experiment_id

client = MlflowClient()

# features and labels
target_column = ["your_target_column"]
X = dataframe.drop(labels=target_column, axis=1)
y = dataframe[target_column]

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# hyper parameters
n_estimators = 10
max_depth = 5

# mlflow tracking
with mlflow.start_run(experiment_id=experiment_id) as run:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # training
    model.fit(X=X_train, y=y_train)

    # predictions
    predict = model.predict(X=X_test)

    log_model_path = "your_log_model_path"
    registered_model_name = "your_registered_model_name"
    run_id = run.info.run_id

    try:
        registered_model = client.get_registered_model(name=registered_model_name)
        registered_model_run_id = registered_model.latest_versions[0].run_id
        registered_model_metrics = client.get_run(run_id=registered_model_run_id).to_dictionary()["data"]["metrics"]

        current_precision = registered_model_metrics["precision_score"]
        current_recall = registered_model_metrics["recall_score"]

        new_precision = precision_score(y_true=y_test, y_pred=predict, average="macro")
        new_recall = recall_score(y_true=y_test, y_pred=predict, average="macro")

        if new_precision > current_precision and new_recall > current_precision:
            # logging hyper parameters
            mlflow.log_param(key="n_estimators", value=n_estimators)
            mlflow.log_param(key="max_depth", value=max_depth)

            # logging metrics
            mlflow.log_metric(
                key="precision_score", 
                value=precision_score(y_true=y_test, y_pred=predict, average="macro")
            )

            mlflow.log_metric(
                key="recall_score",
                value=recall_score(y_true=y_test, y_pred=predict, average="macro")
            )

            # get signature
            signature = mlflow.models.infer_signature(model_input=X_train, model_output=predict)

            # log model
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path=log_model_path, 
                signature=signature,
            )

            # register model with new version
            client.create_model_version(
                name=registered_model_name,
                source=f"runs:/{run_id}/{log_model_path}",
                run_id=run_id
            )

    except mlflow.exceptions.MlflowException:

        # logging hyper parameters
        mlflow.log_param(key="n_estimators", value=n_estimators)
        mlflow.log_param(key="max_depth", value=max_depth)

        # logging metrics

        mlflow.log_metric(
            key="precision_score", 
            value=precision_score(y_true=y_test, y_pred=predict, average="macro")
        )

        mlflow.log_metric(
            key="recall_score",
            value=recall_score(y_true=y_test, y_pred=predict, average="macro")
        )

        # get signature
        signature = mlflow.models.infer_signature(model_input=X_train, model_output=predict)

        # log model
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path=log_model_path, 
            signature=signature,
        )

        # register model
        client.create_registered_model(name=registered_model_name)
        client.create_model_version(
            name=registered_model_name,
            source=f"runs:/{run_id}/{log_model_path}",
            run_id=run_id
        )