import argparse
from audio_augmentation import create_audio_data
from audio_classification import train_gmms, classify_audio
import pandas as pd
from pathlib import Path

from image_classification import classify_images


def merge_scores(classifier_path_list):
    if len(classifier_path_list) > 2:
        raise NotImplementedError("Currently merging of only two classifiers is implemented.")

    names = ["id", "soft", "hard"]
    df1 = pd.read_csv(classifier_path_list[0], delimiter=" ", names=names)
    df2 = pd.read_csv(classifier_path_list[1], delimiter=" ", names=names)

    df_res = pd.merge(df1, df2, on="id")
    df_res["soft"] = (df_res["soft_x"] + df_res["soft_y"]) / 2
    df_res["hard"] = (df_res["soft"] > 0.5).astype(int)
    df_res.to_csv("audio_gmm_image_conv", sep=" ", columns=["id", "soft", "hard"], header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification of images and voices of a person.')
    parser.add_argument('--augment-audio', help='Creates audio augmented data.', action='store_true')
    parser.add_argument('--train-gmm', help='Performs training of GMM models.', action='store_true')
    parser.add_argument('--predict', help='Predict labels for new data.', action='store_true')
    parser.add_argument("--average-classifiers", help='List of files to average and perform new hard classification.',
                        action='append')
    parser.add_argument("--img-data-path", type=Path)
    parser.add_argument("--img-model-path", type=Path, default=Path("models/image_model.pth"))

    args = vars(parser.parse_args())

    if args["augment_audio"]:
        create_audio_data()

    if args["train_gmm"]:
        train_gmms()

    if args["predict"]:
        classify_audio()

    if args["img_data_path"]:
        classify_images(args["img_data_path"], args["img_model_path"])

    if len(args["average_classifiers"]) > 1:
        merge_scores(args["average_classifiers"])
