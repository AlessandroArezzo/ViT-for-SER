import os
import yaml
import argparse

from timm.data import create_loader, resolve_data_config, ImageDataset
from timm.models import create_model

import torch

from parser_csv import ParserCSV
from utils import save_performance, save_confusion_matrix, get_class_labels
import numpy as np
from utils import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import logging
from src import *

# Model to test arguments

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
"""
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
"""
parser = argparse.ArgumentParser(description='PyTorch Timm Tests')
# Dataset / Model parameters
parser.add_argument('--data_eval_dir', metavar='DIR',
                    help='path to validation')
parser.add_argument('--model', metavar='NAME', type=str,
                    help='name model timm')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
# Augmentation & regularization parameters
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
# Misc
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')



# Script arguments

parser.add_argument('model_path', metavar='PATH', type=str,
                    help='model to test relative path')
parser.add_argument('--output_dir', default= "./", metavar='PATH', type=str,
                    help='root outputu dir')
parser.add_argument('--experiment', metavar='NAME', type=str,
                    help='experiment type (research | production | cross-corpus)')
parser.add_argument('--augmentation', metavar='NAME',
                    help='augmentation type (no_aug | time_aug | freq_aug | time_freq_aug)')
parser.add_argument('--pretrained', metavar='NAME', type=str, default='scratch',
                    help='model pretrained (pretrained | scratch)')
parser.add_argument('--data_test_path', metavar='PATH', type=str, default='',
                    help='test data absolute path')
parser.add_argument('--interpolation', metavar='NAME', type=str, default='bicubic',
                    help='resize interpolation')
parser.add_argument('--num_workers', metavar='NAME', type=int, default=4,
                    help='number of workers')
_logger = logging.getLogger('inference')

def _parse_args(config_file):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    args_config.config = config_file

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args = parser.parse_args()
    first_folder = [f.path for f in os.scandir(args.model_path) if f.is_dir()][0]
    config_file = os.path.join(first_folder, "args.yaml")
    args, args_text = _parse_args(config_file)

    root_dir = "./"

    pretrained_str = ""
    if args.pretrained:
        pretrained_str = "pretrained"
    else:
        pretrained_str = "scratch"

    model = ('-').join([args.model, pretrained_str])
    dataset = get_dataset_name_from_path(args.data_test_path)
    num_classes_experiment = get_classes_name_from_path(args.data_test_path)
    aug_type = get_aug_type_from_path(args.data_test_path)
    exp_type = get_exp_type_from_path(args.data_test_path)

    assert exp_type == "within-corpus" or exp_type == "IEMOCAP_train" or exp_type == "cross-corpus"

    output_dir = os.path.join(args.output_dir, exp_type, model, dataset,
                              num_classes_experiment, aug_type)
    class_to_idx = get_class_to_idx(dataset, num_classes_experiment)

    class_labels = get_class_labels(args.data_train_dir)
    class_labels_in_order = {}
    for c in class_labels:
        class_labels_in_order[c] = class_to_idx[c]
    class_labels_in_order = {k: v for k, v in sorted(class_labels_in_order.items(), key=lambda item: item[1])}

    folders = [os.path.basename(f.path) for f in os.scandir(args.data_test_path) if f.is_dir()]

    accuracy_tot, micro_precision_tot, macro_precision_tot, micro_recall_tot, macro_recall_tot, \
    micro_f1_tot, macro_f1_tot = 0, 0, 0, 0, 0, 0, 0
    cm_tot = np.zeros((args.num_classes, args.num_classes))

    for idx, folder in enumerate(folders):
        print("PROCESS FOLDER " + str(idx+1) + "/" + str(len(folders)))
        performance_output = os.path.join(output_dir, folder)

        if not os.path.exists(performance_output):
            os.makedirs(performance_output)

        performance_output_file_path = os.path.join(performance_output, "performance.yaml")
        cm_output_file_path = os.path.join(performance_output, "conf_matrix.png")

        eval_csv = os.path.join(args.data_test_path, folder, "dataset.csv")
        eval_models_dirs = [f.path for f in os.scandir(args.model_path) if f.is_dir()]

        accuracy_folder,  micro_precision_folder, macro_precision_folder, micro_recall_folder, macro_recall_folder,\
        micro_f1_folder, macro_f1_folder = 0, 0, 0, 0, 0, 0, 0
        cm_folder = np.zeros((args.num_classes, args.num_classes))

        for eval_model_dir in eval_models_dirs:

            eval_model_path = os.path.join(eval_model_dir, "model_best.pth.tar")
            if not os.path.exists(eval_csv):
                print("TEST DATASET " + eval_csv + " NOT EXISTS")
                return

            if not os.path.exists(eval_model_path):
                print("MODEL NOT FIND: " + eval_model_path)
                return

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            model = create_model(
                args.model,
                num_classes=args.num_classes,
                checkpoint_path=eval_model_path,
                pretrained=args.pretrained,
                drop_rate=args.drop,
                drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
                drop_path_rate=args.drop_path,
                drop_block_rate=args.drop_block,
                global_pool=args.gp,
                bn_tf=args.bn_tf,
                bn_momentum=args.bn_momentum,
                bn_eps=args.bn_eps,
                scriptable=args.torchscript)

            config = resolve_data_config(vars(args), model=model)
            model.cuda()

            dataset_test = ImageDataset(str(eval_csv), parser=ParserCSV(eval_csv, class_to_idx=class_to_idx))

            loader = create_loader(
                dataset_test,
                input_size=config['input_size'],
                batch_size=1,
                use_prefetcher=True,
                interpolation=config['interpolation'],
                mean=config['mean'],
                std=config['std'],
                num_workers=args.num_workers,
                crop_pct=config['crop_pct'])


            model.eval()

            y_test_true, y_test_predicted = [], []

            with torch.no_grad():
                for batch_idx, (input, target) in enumerate(loader):
                    input = input.cuda()
                    target = target.cuda()
                    labels = model(input)
                    top1 = labels.topk(1)[1].cpu().numpy()

                    y_test_true.append(target.cpu().numpy()[0])
                    y_test_predicted.append(top1[0][0])


            y_test_true = np.array(y_test_true)
            y_test_predicted = np.array(y_test_predicted)

            # calcolo le statistiche
            accuracy_folder += accuracy_score(y_test_true, y_test_predicted)
            micro_precision_folder += precision_score(y_test_true, y_test_predicted, average="micro")
            macro_precision_folder += precision_score(y_test_true, y_test_predicted, average="macro")
            micro_recall_folder += recall_score(y_test_true, y_test_predicted, average="micro")
            macro_recall_folder += recall_score(y_test_true, y_test_predicted, average="macro")
            micro_f1_folder += f1_score(y_test_true, y_test_predicted, average="micro")
            macro_f1_folder += f1_score(y_test_true, y_test_predicted, average="macro")
            #report = classification_report(y_test_true, y_test_predicted)
            cm_folder += confusion_matrix(y_test_true, y_test_predicted, labels=[class_labels_in_order[c] for c in class_labels_in_order],
                                          normalize='true')

            cm_folder[np.isnan(cm_folder)] = 0

        accuracy = accuracy_folder / len(eval_models_dirs)
        micro_precision = micro_precision_folder / len(eval_models_dirs)
        macro_precision = macro_precision_folder / len(eval_models_dirs)
        micro_recall = micro_recall_folder / len(eval_models_dirs)
        macro_recall = macro_recall_folder / len(eval_models_dirs)
        micro_f1 = micro_f1_folder / len(eval_models_dirs)
        macro_f1 = macro_f1_folder / len(eval_models_dirs)

        cm = cm_folder / len(eval_models_dirs)
        sum_of_rows = cm.sum(axis=1)
        cm = cm / sum_of_rows[:, np.newaxis]
        cm[np.isnan(cm)] = 0

        model_stats = {"accuracy": str(accuracy * 100),
                       "micro_precision": str(micro_precision * 100),
                       "micro_recall": str(micro_recall * 100),
                       "micro_f1": str(micro_f1 * 100),
                       "macro_precision": str(macro_precision * 100),
                       "macro_recall": str(macro_recall * 100),
                       "macro_f1": str(macro_f1 * 100), }

        accuracy_tot += accuracy
        micro_precision_tot += micro_precision
        macro_precision_tot += macro_precision
        micro_recall_tot += micro_recall
        macro_recall_tot += macro_recall
        micro_f1_tot += micro_f1
        macro_f1_tot += macro_f1

        cm_tot += cm

        print(model_stats)

        save_performance(model_stats, performance_output_file_path)
        print("Performance folder " + str(len(folders)) + " saved on: " + performance_output_file_path)

        save_confusion_matrix(cm, class_labels_in_order, cm_output_file_path)
        print("Confusion matrix model saved on: " + cm_output_file_path)

    accuracy = accuracy_tot / len(folders)
    micro_precision = micro_precision_tot / len(folders)
    micro_recall = macro_precision_tot / len(folders)
    micro_f1 = micro_recall_tot / len(folders)
    macro_precision = macro_recall_tot / len(folders)
    macro_recall = micro_f1_tot / len(folders)
    macro_f1 = macro_f1_tot / len(folders)

    cm = cm_tot / len(folders)
    sum_of_rows = cm.sum(axis=1)
    cm = cm / sum_of_rows[:, np.newaxis]

    model_stats = {"accuracy": str(accuracy * 100),
                   "micro_precision": str(micro_precision * 100),
                   "micro_recall": str(micro_recall * 100),
                   "micro_f1": str(micro_f1 * 100),
                   "macro_precision": str(macro_precision * 100),
                   "macro_recall": str(macro_recall * 100),
                   "macro_f1": str(macro_f1 * 100), }

    performance_output_file_path = os.path.join(output_dir, "performance.yaml")
    save_performance(model_stats, performance_output_file_path)
    print("Performance model saved on: " + performance_output_file_path)

    cm_output_file_path = os.path.join(output_dir, "conf_matrix.png")
    save_confusion_matrix(cm, class_labels_in_order, cm_output_file_path)
    print("Confusion matrix model saved on: " + cm_output_file_path)

if __name__ == '__main__':
    main()