import os

import yaml
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn


def update_graph(train_loss_history, val_loss_history, val_acc_history, path):
    """
    Method that update graphs with the trend of loss and accuracy on the train and validation data
    :param train_loss_history, val_loss_history: list that contains loss value for train and validation detected
                                                 at each epoch
           train_acc_history, val_loss_hystory: list that contains mean accuracy for train and validation
                                                detected at each epoch
           path: path of the directory where to save the graphs
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)

    losses_img_file = os.path.join(path, "training_losses.png")
    acc_img_file = os.path.join(path, "training_accuracy.png")
    epochs = np.arange(1, len(train_loss_history) + 1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Plot Training/Validation Losses")
    plt.ylim(0, max(max(train_loss_history), max(val_loss_history)))
    plt.plot(epochs, train_loss_history, label="average train loss")
    plt.plot(epochs, val_loss_history, label="average validation loss")
    plt.legend()
    plt.savefig(losses_img_file)
    plt.close()
    plt.title("Plot Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.ylim(0, 100)
    # plt.plot(epochs, train_acc_history, label="average train accuracy")
    plt.plot(epochs, val_acc_history, label="average validation accuracy")
    plt.legend()
    plt.savefig(acc_img_file)
    plt.close()


def read_history_from_csv(path):
    csv_path = os.path.join(path, "summary.csv")
    train_loss_history, val_loss_history, val_acc_history = [], [], []
    if not os.path.exists(csv_path):
        print("ERROR: not find csv in path ", csv_path)
        return
    with open(csv_path, mode='r') as file:
        for idx, line in enumerate(csv.reader(file)):
            if line[0] != "epoch":
                train_loss_history.append(float(line[1]))
                val_loss_history.append(float(line[2]))
                val_acc_history.append(float(line[3]))
    return train_loss_history, val_loss_history, val_acc_history


def save_confusion_matrix(cm, labels, fname):
    df_cm = pd.DataFrame(cm, index=[str(i) for i in labels],
                         columns=[str(i) for i in labels])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.savefig(fname, dpi=240)
    plt.close()


def save_performance(performance, path):
    with open(path, 'w') as outfile:
        yaml.dump(performance, outfile, default_flow_style=False)


def get_emotion(filename):
    head, tail = os.path.split(filename)
    tail_split = tail.split("_")
    return tail_split[0]


def get_class_to_idx(dataset, num_classes):
    class_to_idx = {}
    if num_classes == "3_classes":
        class_to_idx = {'neutral': 0, 'positive': 1, 'negative': 2}
    elif num_classes == "4_classes" or num_classes == "4_classes_excited":
        class_to_idx = {'anger': 0, 'happy': 1, 'neutral': 2, "sad": 3}
    elif num_classes == "all_classes":
        if dataset == "EMODB":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'neutral': 5}
        elif dataset == "DEMOS":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'surprised': 5}
        elif dataset == "IEMOCAP":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'surprised': 6, 'neutral': 5,
                            'excited': 7, 'frustration': 8}
        else:
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'neutral': 5, 'surprised': 6}
    return class_to_idx


def get_speaker_class_to_idx():
    class_to_idx = {'demos-01': 0, 'demos-02': 1, 'demos-03': 2, 'demos-04': 3,
                    'demos-05': 4, 'emovo-01': 5, 'emovo-02': 6, 'emovo-03': 7, 'emovo-04': 8,
                    'emovo-05': 9, 'emovo-06': 10}
    return class_to_idx

def get_gender_class_to_idx():
    class_to_idx = {'F': 0, 'M': 1}
    return class_to_idx

def get_corpus_class_to_idx():
    class_to_idx = {"emovo": 0, "iemocap": 1, "ravdess": 2, "savee": 3, "emodb": 4, 'demos':5 }
    return class_to_idx

def get_class_labels(train_dataset_path):
    classes_label = []
    for root, subdirs, files in os.walk(train_dataset_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() == ".csv":
                dataset_csv = open(os.path.join(root, file))
                csvreader = csv.reader(dataset_csv)
                for idx, row in enumerate(csvreader):
                    if idx > 0:
                        current_class = row[1]
                        if current_class not in classes_label and current_class != "class":
                            classes_label.append(current_class)
    return classes_label


def check_read_wav_from_model_name(model_name):
    first_part_name = model_name.split('_')[0]
    if first_part_name == 'speaker':
        return True
    return False

def get_dataset_name_from_path(path):
    all_datasets = ["EMOVO", "IEMOCAP", "RAVDESS", "SAVEE", "EMODB", "DEMOS", "TESS"]
    path_tails = path.split('/')
    for dataset in all_datasets:
        if dataset in path_tails:
            return dataset
    return None

def get_classes_name_from_path(path):
    all_classes_name = ["3_classes", "3_classes_excited", "4_classes", "4_classes_excited", "all_classes",
                        "all_classes_excited"]
    path_tails = path.split('/')
    for classes_name in all_classes_name:
        if classes_name in path_tails:
            return classes_name
    return None

def get_aug_type_from_path(path):
    all_aug = ["no_aug", "balanced_aug", "undersampling", "undersampling_balanced", "undersampling_aug",
               "undersampling_aug_balanced", "time_aug", "freq_aug", "time_freq_aug"]
    path_tails = path.split('/')
    for aug in all_aug:
        if aug in path_tails:
            return aug
    return None

def get_exp_type_from_path(path):
    all_exp = ["within-corpus", "IEMOCAP_train", "cross-corpus", "cross-corpus-demos", "cross-corpus-men",
               "cross-corpus-women"]
    path_tails = path.split('/')
    for exp in all_exp:
        if exp in path_tails:
            return exp
    return None


