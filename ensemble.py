import click
import os
import torch
import pdb
import streamlit as st
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dolos.predict import (
    PREDICT_CONFIGS,
    load_model,
    save_pred_as_img,
    get_predictions_path
)
from dolos.train_full_supervision import (
    get_mask_size,
    CONFIGS
)
from mesonet.classifiers import (
    Meso4
)
from mesonet.pipeline import *

@click.command()
@click.option("-s", "supervision", type=click.Choice(["full", "weak"]), required=True)
@click.option("-t", "train_config_name", required=True)
@click.option("-p", "predict_config_name", required=True)
@click.option("-v", "to_visualize", is_flag=True, default=False)
@click.option("--save-images", "to_save_images", is_flag=True, default=False)
def main(
    supervision,
    train_config_name,
    predict_config_name,
    to_visualize=False,
    to_save_images=False,
):
    train_config = CONFIGS[train_config_name]
    dataset = PREDICT_CONFIGS[predict_config_name]["dataset"]
    device = "cuda"

    method_name = "patch-forensics"
    patch_model = load_model(supervision, train_config_name, train_config, device)

    num_images = len(dataset)
    mask_size = get_mask_size(train_config)
    # patch_mask_preds = np.zeros((num_images,) + mask_size)
    patch_label_preds = np.zeros((num_images,))

    mesonet_classifier = Meso4()
    mesonet_classifier.load('MesoNet/weights/Meso4_DF.h5')
    mesonet_label_preds = np.zeros((num_images,))

    ensemble_label_preds = np.zeros((num_images,))

    load_image = train_config["load-image"]

    out_dir = os.path.join(
        "output",
        method_name,
        "{}-supervision".format(supervision),
        "predictions-train-config-{}-predict-config-{}".format(
            train_config_name, predict_config_name
        ),
    )

    for i in tqdm(range(num_images)):
        image = load_image(dataset, i, split="valid")

        # Dolos patch classifier prediction

        with torch.no_grad():
            image = image.to(device)
            mask_pred = patch_model(image.unsqueeze(0))

        mask_pred = F.softmax(mask_pred, dim=1)
        mask_pred = mask_pred[0, 1]
        mask_pred = mask_pred.detach().cpu().numpy()

        # patch_mask_preds[i] = mask_pred
        patch_label_preds[i] = np.mean(mask_pred)

        # Mesonet prediction

        mesonet_pred = mesonet_classifier.predict(image)
        mesonet_label_preds[i] = mesonet_pred[0][0]

        # generate ensemble pred (0 if either model predicts 0/deepfake)
        if (np.mean(mask_pred) <= 0.5 or mesonet_pred[0][0] <= 0.5):
            ensemble_label_preds[i] = 0
        else:
            ensemble_label_preds[i] = 1

        # save image out
        if to_save_images:
            save_pred_as_img(dataset, i, mask_pred, out_dir)

        if to_visualize:
            fig, axs = plt.subplots(nrows=1, ncols=2)

            mask_true = dataset.load_mask(i)

            axs[0].imshow(mask_true)
            axs[0].set_title("true")
            axs[0].set_axis_off()

            axs[1].imshow(mask_pred)
            axs[1].set_title("pred")
            axs[1].set_axis_off()

            st.pyplot(fig)
            pdb.set_trace()

    output_path = get_predictions_path(
        method_name, supervision, train_config_name, predict_config_name
    )
    np.savez(output_path, ensemble_label_preds=ensemble_label_preds, patch_label_preds=patch_label_preds, mesonet_label_preds=mesonet_label_preds)


if __name__ == "__main__":
    main()
