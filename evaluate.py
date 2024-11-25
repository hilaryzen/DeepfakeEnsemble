import click
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dolos.predict import get_predictions_path


@click.command()
@click.option("-s", "supervision", type=click.Choice(["full", "weak"]), required=True)
@click.option("-t", "train_config_name", required=True)
@click.option("-r", "dataset_name_real", required=True)
@click.option("-d", "dataset_name_deepfake", required=True)
@click.option("-v", "to_visualize", is_flag=True, default=False)
def main(supervision, train_config_name, dataset_name_real, dataset_name_deepfake, to_visualize=False):
    method_name = "patch-forensics"
    print(supervision, train_config_name, dataset_name_real, dataset_name_deepfake)
    pred_real = np.load(get_predictions_path(method_name, supervision, train_config_name, dataset_name_real))["ensemble_label_preds"]
    pred_fake = np.load(get_predictions_path(method_name, supervision, train_config_name, dataset_name_deepfake))["ensemble_label_preds"]

    num_real = len(pred_real)
    num_fake = len(pred_fake)

    print("Accuracy for real images: ", accuracy_score(np.ones(num_real), pred_real))
    print("Accuracy for fake images: ", accuracy_score(np.zeros(num_fake), pred_fake))

    true = np.hstack((np.zeros(num_fake), np.ones(num_real)))
    pred = np.hstack((pred_fake, pred_real))

    metrics = {
        "accuracy": accuracy_score(true, pred), 
        "precision": precision_score(true, pred),
        "recall": recall_score(true, pred),
        "f1": f1_score(true, pred)
    }
    print(metrics)


if __name__ == "__main__":
    main()
