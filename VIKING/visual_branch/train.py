import argparse
import os
import shutil
from typing import Optional
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn
from argparse import Namespace

import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.datasets as datasets

# task specific package
import models.dual_model as models  # DL_Conv, DL_FC, DL_ResNet, Dual_Model, Dual_Model_Baseline, Dual_Model_MLB
import dual_model.lib.engine_v2 as engine

import pdb
import neptune
import getpass, socket
import sys
from datetime import datetime

os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = "true"

model_names = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)


def add_time_stamp(log_dir: str) -> str:
    time_stamp: str = datetime.now().isoformat()
    return os.path.join(log_dir, time_stamp)


parser = argparse.ArgumentParser(
    description="Train/Evaluate models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
##################################################
#  yaml options file contains all default choices #
parser.add_argument(
    "--path_opt",
    default="options/dual_model/default.yaml",
    type=str,
    help="path to a yaml options file",
)
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument("--dir_logs", type=add_time_stamp, help="dir logs")
# data options
parser.add_argument("--vqa_trainsplit", type=str, choices=["train", "trainval"])
# model options
parser.add_argument(
    "--arch",
    choices=model_names,
    help="vqa model architecture: " + " | ".join(model_names),
)
parser.add_argument(
    "-lr", "--learning_rate", type=float, help="initial learning rate"
)
parser.add_argument(
    "-wdecay", "--weight_decay", type=float, help="weight decay"
)
parser.add_argument("-b", "--batch_size", type=int, help="mini-batch size")
parser.add_argument("--epochs", type=int, help="number of total epochs to run")
parser.add_argument(
    "--eval_epochs",
    type=int,
    default=10,
    help="Number of epochs to evaluate the model",
)
# options not in yaml file
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--resume", default="", type=str, help="path to latest checkpoint"
)
parser.add_argument(
    "--save_model",
    default=True,
    type=utils.str2bool,
    help="able or disable save model and optim state",
)
parser.add_argument(
    "--save_all_from",
    type=int,
    help="""delete the preceding checkpoint until an epoch,"""
    """ then keep all (useful to save disk space)')""",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation and test set",
)
parser.add_argument(
    "--print_freq", "-p", default=1000, type=int, help="print frequency"
)
################################################
parser.add_argument(
    "-ho",
    "--help_opt",
    dest="help_opt",
    action="store_true",
    help="show selected options before running",
)
parser.add_argument(
    "--beam_search",
    action="store_true",
    help="whether to use beam search, the batch_size will be set to 1 automatically",
)

parser.add_argument(
    "--dual_training",
    action="store_true",
    help="Whether to use additional loss",
)

parser.add_argument(
    "--share_embeddings",
    action="store_true",
    help="Whether to share the embeddings",
)

# parser.add_argument('--finetuning_conv_epoch', type=int, default=10, help='From which epoch to finetuning the conv layers')

parser.add_argument(
    "--alternative_train",
    type=float,
    default=-1.0,
    help="The sample rate for QG training. if [alternative_train] > 1 or < 0, then jointly train.",
)

parser.add_argument(
    "--partial",
    type=float,
    default=-1.0,
    help="Only use part of the VQA dataset. Valid range is (0, 1). [default: -1.]",
)

parser.add_argument(
    "--neptune_on",
    action="store_true",
    help="Whether to upload results to neptune",
)


def construct_neptune_experiment(
    properties: dict = {}, parameters: dict = {}
) -> neptune.experiments.Experiment:
    properties.update(
        {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "cmd": " ".join(sys.argv),
        }
    )
    neptune.init("artQA/artQA-IJICAI-submission")
    neptune_exp = neptune.create_experiment(
        name=f"train-iQAN",
        params=parameters,
        properties=properties,
        tags=["question-image", "iQAN"],
        upload_stdout=True,
    )
    return neptune_exp


def construct_dataloader(
    split: str, options: dict
) -> torch.utils.data.DataLoader:

    dataset = datasets.factory_VQA(
        options["dataset"][split], split, options["dataset"]
    )

    shuffle = split == "train"
    dataset_loader = dataset.data_loader(
        batch_size=options["optim"]["batch_size"],
        num_workers=1,
        shuffle=shuffle,
    )
    return dataset_loader


def make_log_dir(log_dir: str) -> None:
    if os.path.isdir(log_dir):
        if click.confirm(
            "Logs directory already exists in {}. Erase?".format(
                log_dir, default=False
            )
        ):
            os.system(f"rm -r {log_dir}")
        else:
            return
    os.system(f"mkdir -p {log_dir}")


def log_options(log_dir: str, args: Namespace, options: dict) -> None:
    path_new_opt = os.path.join(log_dir, os.path.basename(args.path_opt))
    path_args = os.path.join(log_dir, "args.yaml")
    with open(path_new_opt, "w") as f:
        yaml.dump(options, f, default_flow_style=False)
    with open(path_args, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def construct_logger(options: dict) -> logger.Experiment:
    exp_name = os.path.basename(options["logs"]["dir_logs"])  # add timestamp
    exp_logger = logger.Experiment(exp_name, options)
    exp_logger.add_meters("train", make_meters())
    # exp_logger.add_meters("test", make_meters())
    exp_logger.add_meters("val", make_meters())
    return exp_logger


def training_loop(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    exp_logger: logger.Experiment,
    print_freq: int,
    start_epoch: int,
    dual_training: bool,
    alternative_train: float,
    options: dict,
    save_model: bool = True,
    save_all_from: Optional[int] = None,
    neptune_exp: Optional[neptune.experiments.Experiment] = None,
) -> float:

    best_acc1: float = 0.0
    best_acc5: float = 0.0
    best_acc10: float = 0.0
    best_loss_q: float = 1000.0

    for epoch in range(start_epoch, options["optim"]["epochs"]):
        engine.train(
            train_loader,
            model,
            optimizer,
            exp_logger,
            epoch,
            print_freq,
            dual_training=dual_training,
            alternative_train=alternative_train,
            neptune_exp=neptune_exp,
        )

        # evaluate on validation set
        acc1, acc5, acc10, loss_q = engine.validate(
            val_loader, model, exp_logger, epoch, print_freq
        )

        if neptune_exp is not None:
            neptune_exp.log_metric("Acc@1/val", epoch, acc1)
            neptune_exp.log_metric("Acc@5/val", epoch, acc5)
            neptune_exp.log_metric("Acc@10/val", epoch, acc10)
            neptune_exp.log_metric("Loss_Q/val", epoch, loss_q)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        is_best_q = loss_q < best_loss_q
        best_acc5 = acc5 if is_best else best_acc5
        best_acc10 = acc10 if is_best else best_acc10
        best_acc1 = acc1 if is_best else best_acc1
        best_loss_q = loss_q if is_best_q else best_loss_q

        print(
            "** [Best]\tAcc@1: {0:.2f}%\tAcc@5: {1:.2f}%\tAcc@10: {2:.2f}% \tQ_Loss: {3:.4f}".format(
                best_acc1, best_acc5, best_acc10, best_loss_q
            )
        )
        save_checkpoint(
            {
                "epoch": epoch,
                "arch": options["model"]["arch"],
                "best_acc1": best_acc1,
                "best_acc5": best_acc5,
                "best_acc10": best_acc10,
                "exp_logger": exp_logger,
            },
            model.module.state_dict(),  # model.module.state_dict(),
            optimizer.state_dict(),
            options["logs"]["dir_logs"],
            save_model,
            save_all_from,
            is_best,
            is_best_q,
        )

    return best_acc1


def main():
    args = parser.parse_args()

    if args.path_opt is not None:
        with open(args.path_opt, "r") as handle:
            options = yaml.load(handle)

    if "model" not in options.keys():
        options["model"] = {}

    if args.dual_training:
        options["logs"]["dir_logs"] += "_dual_training"

    print("## args")
    pprint(vars(args))
    print("## options")
    pprint(options)

    if args.neptune_on:
        parameters = {"log_dir": options["logs"]["dir_logs"]}
        neptune_exp = construct_neptune_experiment(parameters=parameters)

        for split in ["train", "val", "test"]:
            neptune_exp.log_text(
                f"{split}_data_version", md5(f"data/artvqa/{split}.json")
            )

    if args.help_opt:
        return

    # Set datasets
    print("Loading dataset....")

    train_loader = construct_dataloader("train", options)

    val_loader = construct_dataloader("val", options)

    print("Done.")
    print("Setting up the model...")

    model = getattr(models, options["model"]["arch"])(
        options["model"],
        train_loader.dataset.vocab_words(),
        train_loader.dataset.vocab_answers(),
    )

    if args.share_embeddings:
        model.set_share_parameters()

    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=options["optim"]["lr"],
        weight_decay=options["optim"]["weight_decay"],
    )

    model = nn.DataParallel(model).cuda()

    # Optionally resume from a checkpoint
    exp_logger = None
    if args.resume:
        print("Loading saved model...")
        args.start_epoch, best_acc1, exp_logger = load_checkpoint(
            model,
            optimizer,  # model.module, optimizer,
            os.path.join(options["logs"]["dir_logs"], args.resume),
        )
    else:
        log_dir = options["logs"]["dir_logs"]
        make_log_dir(log_dir)
        log_options(log_dir, args, options)

        if args.neptune_on:
            for option_logs in os.listdir(log_dir):
                neptune_exp.log_artifact(os.path.join(log_dir, option_logs))

    if exp_logger is None:
        exp_logger = construct_logger(options)

    print("Start training")

    training_loop(
        train_loader,
        val_loader,
        model,
        optimizer,
        exp_logger,
        args.print_freq,
        args.start_epoch,
        args.dual_training,
        args.alternative_train,
        options,
        args.save_model,
        args.save_all_from,
        neptune_exp=neptune_exp if args.neptune_on else None,
    )

    if args.neptune_on:
        neptune_exp.stop()


def make_meters():
    meters_dict = {
        "loss": logger.AvgMeter(),
        "loss_a": logger.AvgMeter(),
        "loss_q": logger.AvgMeter(),
        "batch_time": logger.AvgMeter(),
        "data_time": logger.AvgMeter(),
        "epoch_time": logger.SumMeter(),
        "bleu_score": logger.AvgMeter(),
        "acc1": logger.AvgMeter(),
        "acc5": logger.AvgMeter(),
        "acc10": logger.AvgMeter(),
        "dual_loss": logger.AvgMeter(),
    }
    return meters_dict


def save_results(
    results, epoch, split_name, dir_logs, dir_vqa, is_testing=True
):
    if is_testing:
        subfolder_name = "evaluate"
    else:
        subfolder_name = "epoch_" + str(epoch)
    dir_epoch = os.path.join(dir_logs, subfolder_name)
    name_json = "SemArt_Prediction.json"
    # TODO: simplify formating
    if "test" in split_name:
        name_json = "vqa_" + name_json
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system("mkdir -p " + dir_epoch)
    with open(path_rslt, "w") as handle:
        json.dump(results, handle)


def save_checkpoint(
    info,
    model,
    optim,
    dir_logs,
    save_model,
    save_all_from=None,
    is_best=True,
    is_best_q=True,
):
    os.system("mkdir -p " + dir_logs)
    if save_all_from is None:
        path_ckpt_info = os.path.join(dir_logs, "ckpt_info.pth.tar")
        path_ckpt_model = os.path.join(dir_logs, "ckpt_model.pth.tar")
        path_ckpt_optim = os.path.join(dir_logs, "ckpt_optim.pth.tar")
        path_best_info = os.path.join(dir_logs, "best_info.pth.tar")
        path_best_model = os.path.join(dir_logs, "best_model.pth.tar")
        path_best_optim = os.path.join(dir_logs, "best_optim.pth.tar")
        path_best_info_q = os.path.join(dir_logs, "best_info_VQG.pth.tar")
        path_best_model_q = os.path.join(dir_logs, "best_model_VQG.pth.tar")
        path_best_optim_q = os.path.join(dir_logs, "best_optim_VQG.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        if is_best_q:
            shutil.copyfile(path_ckpt_info, path_best_info_q)
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
            if is_best_q:
                shutil.copyfile(path_ckpt_model, path_best_model_q)
                shutil.copyfile(path_ckpt_optim, path_best_optim_q)
    else:
        is_best = False  # because we don't know the test accuracy
        path_ckpt_info = os.path.join(dir_logs, "ckpt_epoch,{}_info.pth.tar")
        path_ckpt_model = os.path.join(dir_logs, "ckpt_epoch,{}_model.pth.tar")
        path_ckpt_optim = os.path.join(dir_logs, "ckpt_epoch,{}_optim.pth.tar")
        # save info & logger
        path_logger = os.path.join(dir_logs, "logger.json")
        info["exp_logger"].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info["epoch"]))
        #  save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info["epoch"]))
            torch.save(optim, path_ckpt_optim.format(info["epoch"]))
        if info["epoch"] > 1 and info["epoch"] < save_all_from + 1:
            os.system("rm " + path_ckpt_info.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_model.format(info["epoch"] - 1))
            os.system("rm " + path_ckpt_optim.format(info["epoch"] - 1))
    if not save_model:
        print("Warning train.py: checkpoint not saved")


def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info = path_ckpt + "_info.pth.tar"
    path_ckpt_model = path_ckpt + "_model.pth.tar"
    path_ckpt_optim = path_ckpt + "_optim.pth.tar"
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc1 = 0
        exp_logger = None
        if "epoch" in info:
            start_epoch = info["epoch"]
        else:
            print("Warning train.py: no epoch to resume")
        if "best_acc1" in info:
            best_acc1 = info["best_acc1"]
        else:
            print("Warning train.py: no best_acc1 to resume")
        if "exp_logger" in info:
            exp_logger = info["exp_logger"]
        else:
            print("Warning train.py: no exp_logger to resume")
    else:
        print(
            "Warning train.py: no info checkpoint found at '{}'".format(
                path_ckpt_info
            )
        )
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print(
            "Warning train.py: no model checkpoint found at '{}'".format(
                path_ckpt_model
            )
        )
    print(
        "=> loaded checkpoint '{}' (epoch {}, best_acc1 {})".format(
            path_ckpt, start_epoch, best_acc1
        )
    )
    return start_epoch, best_acc1, exp_logger


import hashlib


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    main()

