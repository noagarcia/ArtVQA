from typing import Dict, List, Optional, Tuple
import argparse
import os
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn

import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.datasets as datasets

# task specific package
import models.dual_model as models
import dual_model.lib.engine_v2 as engine

import pdb

model_names: List[str] = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)


def main():
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
    parser.add_argument("--dir_logs", type=str, help="dir logs")
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
        "--dual_training", action="store_true", help="Whether to use additional loss"
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

    args = parser.parse_args()

    # Set options
    options = {
        "vqa": {"trainsplit": args.vqa_trainsplit, "partial": args.partial},
        "logs": {"dir_logs": args.dir_logs},
        "optim": {
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "eval_epochs": args.eval_epochs,
        },
        "model": {},
    }

    if args.path_opt is not None:

        with open(args.path_opt, "r") as handle:
            options_yaml = yaml.load(handle)

        options = utils.update_values(options, options_yaml)

    if args.dual_training:
        options["logs"]["dir_logs"] += "_dual_training"

    print("## args")
    pprint(vars(args))
    print("## options")
    pprint(options)

    if args.help_opt:
        return

    best_acc1: float = 0.0
    best_acc5: float = 0.0
    best_acc10: float = 0.0
    best_loss_q: float = 1000.0

    # Set datasets
    print("Loading dataset....")
    testset = datasets.factory_VQA(
        "test",
        options["vqa"],
        opt_coco=options.get("coco", None),
        opt_clevr=options.get("clevr", None),
        opt_vgenome=options.get("vgnome", None),
    )

    test_loader = testset.data_loader(
        batch_size=1 if args.beam_search else options["optim"]["batch_size"],
        num_workers=1,
    )

    print("Done.")
    print("Setting up the model...")
    # pdb.set_trace()
    # Set model, criterion and optimizer
    # assert options['model']['arch_resnet'] == options['coco']['arch'], 'Two [arch] should be set the same.'
    model = getattr(models, options["model"]["arch"])(
        options["model"], testset.vocab_words(), testset.vocab_answers()
    )

    if args.share_embeddings:
        model.set_share_parameters()

    # Optionally resume from a checkpoint
    exp_logger = None
    if args.resume:
        print("Loading saved model...")

        args.start_epoch, _, exp_logger = load_checkpoint(
            os.path.join(options["logs"]["dir_logs"], args.resume),
            model,
            optimizer=None,  # model.module, optimizer,
        )
    else:
        # Or create logs directory
        if os.path.isdir(options["logs"]["dir_logs"]):
            if click.confirm(
                "Logs directory already exists in {}. Erase?".format(
                    options["logs"]["dir_logs"], default=False
                )
            ):
                os.system("rm -r " + options["logs"]["dir_logs"])
            else:
                return
        os.system("mkdir -p " + options["logs"]["dir_logs"])
        path_new_opt = os.path.join(
            options["logs"]["dir_logs"], os.path.basename(args.path_opt)
        )
        path_args = os.path.join(options["logs"]["dir_logs"], "args.yaml")
        with open(path_new_opt, "w") as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if exp_logger is None:
        #  Set loggers
        exp_name = os.path.basename(options["logs"]["dir_logs"])  # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters("test", make_meters())
        if options["vqa"]["trainsplit"] == "train":
            exp_logger.add_meters("val", make_meters())
        exp_logger.info["model_params"] = utils.params_count(model)
        print("Model has {} parameters".format(exp_logger.info["model_params"]))

    # Add OoV answer embedding
    model.answer_embeddings = add_oov_emb(model.answer_embeddings)

    #  Begin evaluation and training
    model = nn.DataParallel(model).cuda()
    if args.evaluate:
        print("Start evaluating...")

        evaluate_result = engine.evaluate(
            test_loader, model, exp_logger, args.print_freq
        )

        save_results(
            evaluate_result,
            args.start_epoch,
            testset.split_name(),
            options["logs"]["dir_logs"],
            options["vqa"]["dir"],
        )

        return


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
    results: List[Dict],
    epoch: int,
    split_name: str,
    dir_logs: str,
    dir_vqa: str,
    is_testing: bool = True,
) -> None:
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


def load_checkpoint(
    path_ckpt: str,
    model: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[int, float, Optional[logger.Experiment]]:
    path_ckpt_info = path_ckpt + "_info.pth.tar"
    path_ckpt_model = path_ckpt + "_model.pth.tar"
    start_epoch: int = 0
    best_acc1: float = 0
    exp_logger = None
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
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
            "Warning train.py: no info checkpoint found at '{}'".format(path_ckpt_info)
        )
        raise RuntimeError(f"{path_ckpt_info} not found")

    if model is not None:
        if os.path.isfile(path_ckpt_model):
            model_state = torch.load(path_ckpt_model)
            model.load_state_dict(model_state)
            print(
                f"=> loaded checkpoint '{path_ckpt}' (epoch {start_epoch}, best_acc1 {best_acc1})"
            )
        else:
            print(
                "Warning train.py: no model checkpoint found at '{}'".format(
                    path_ckpt_model
                )
            )
            raise RuntimeError(f"{path_ckpt_model} not found")

    return start_epoch, best_acc1, exp_logger


def add_oov_emb(src_emb: nn.Embedding) -> nn.Embedding:
    n_vocab = src_emb.num_embeddings
    emb_dim = src_emb.embedding_dim
    weight = torch.cat([src_emb.weight, src_emb.weight.mean(0, keepdim=True)], dim=0)
    new_emb = nn.Embedding(n_vocab + 1, emb_dim, _weight=weight, padding_idx=n_vocab)
    return new_emb


if __name__ == "__main__":
    main()
