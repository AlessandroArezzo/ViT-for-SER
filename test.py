import os
import yaml
import argparse

from timm.data import create_loader, resolve_data_config, create_dataset
from timm.models import create_model

import torch
from utils import save_performance, save_confusion_matrix
import numpy as np

from src import cct_7_3x1_32_c100

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import logging

# Model to test arguments

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
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

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
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
    args, args_text = _parse_args()

    root_dir = "./"

    pretrained_str = ""
    if args.pretrained:
        pretrained_str = "pretrained"
    else:
        pretrained_str = "scratch"

    model = ('-').join([args.model, pretrained_str])
    output_dir = os.path.join(root_dir, "test", args.experiment, model, args.augmentation)

    test_dataset = ""
    if args.experiment == "production" or args.experiment == "cross-corpus":
        test_dataset = os.path.basename(os.path.normpath(args.data_test_path))
        output_dir = os.path.join(output_dir, test_dataset)

    #emo2lab = {"anger": 3, "disgust": 4, "fearful": 5, "happy": 0, "sad": 01, "surprised": 6, "neutral": 2}
    # emo2lab={"positive":0, "negative":01, "neutral":2}

    performance_output_file_path = os.path.join(output_dir, "performance.yaml")
    cm_output_path = os.path.join(output_dir, "conf_matrix.png")

    if not os.path.exists(args.data_test_path):
        print("TEST DIR " + args.data_test_path + " NOT EXISTS")
        return

    if not os.path.exists(args.model_path):
        print("MODEL NOT FIND: " + args.model_path)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("LOADING MODEL...")

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        checkpoint_path=args.model_path,
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

    print("LOADING DATA...")

    dataset_test = create_dataset(
        args.dataset, root=args.data_test_path, is_training=False, batch_size=1)

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

    emo2label = dataset_test.parser.class_to_idx

    print("TEST MODEL...")

    model.eval()

    y_test_true, y_test_predicted = [], []

    #top1_m = AverageMeter()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            labels = model(input)
            top1 = labels.topk(1)[1].cpu().numpy()

            y_test_true.append(target.cpu().numpy()[0])
            y_test_predicted.append(top1[0][0])

            #acc1, acc5 = acc(labels, target, topk=(01, 5))
            #top1_m.update(acc1.item(), labels.size(0))

    y_test_true = np.array(y_test_true)
    y_test_predicted = np.array(y_test_predicted)

    # calcolo le statistiche
    accuracy = accuracy_score(y_test_true, y_test_predicted)
    micro_precision = precision_score(y_test_true, y_test_predicted, average="micro")
    macro_precision = precision_score(y_test_true, y_test_predicted, average="macro")
    micro_recall = recall_score(y_test_true, y_test_predicted, average="micro")
    macro_recall = recall_score(y_test_true, y_test_predicted, average="macro")
    micro_f1 = f1_score(y_test_true, y_test_predicted, average="micro")
    macro_f1 = f1_score(y_test_true, y_test_predicted, average="macro")
    #report = classification_report(y_test_true, y_test_predicted)
    cm = confusion_matrix(y_test_true, y_test_predicted, normalize='true')

    model_stats = {}

    if args.experiment == "production" or args.experiment == "cross-corpus":
        model_stats["TEST DATASET"] = test_dataset

    model_stats = {"accuracy": str(accuracy * 100),
                   "micro_precision": str(micro_precision * 100),
                   "micro_recall": str(micro_recall * 100),
                   "micro_f1": str(micro_f1 * 100),
                   "macro_precision": str(macro_precision * 100),
                   "macro_recall": str(macro_recall * 100),
                   "macro_f1": str(macro_f1 * 100), }

    print(model_stats)

    save_performance(model_stats, performance_output_file_path)
    print("Performance model saved on: " + performance_output_file_path)

    save_confusion_matrix(cm, emo2label.keys(), cm_output_path)
    print("Confusion matrix model saved on: " + cm_output_path)

    """
    test_images = []
    for path, subdirs, files in os.walk(args.data_test_path):
      for image in files:
        test_images.append(os.path.join(args.data_test_path, get_emotion(image), image))

    model.eval()
    y_true, y_pred = [], []
    """

    #print("ACCURACY AVERAGE METER: "+str(top1_m.avg))


if __name__ == '__main__':
    main()