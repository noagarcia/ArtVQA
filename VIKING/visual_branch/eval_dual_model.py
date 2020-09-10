from typing import List, Dict
import argparse
import os
import yaml
import json
from pprint import pprint

import torch
import torch.nn as nn
from argparse import Namespace

import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.datasets as datasets

# task specific package
import models.dual_model as models
import dual_model.lib.engine_v2 as engine

import pdb
import neptune
import getpass
import socket
import sys

os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = "true"

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
    parser.add_argument("dataset_path", type=str, help="dataset file to evaluate")
    parser.add_argument(
        "--path_opt",
        default="options/dual_model/default.yaml",
        type=str,
        help="path to a yaml options file",
    )
    parser.add_argument("--dir_logs", type=str, help="directory of leaned model")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=None, help="mini-batch size"
    )
    parser.add_argument(
        "-ho",
        "--help_opt",
        dest="help_opt",
        action="store_true",
        help="show selected options before running",
    )
    parser.add_argument(
        "--share_embeddings",
        action="store_true",
        help="Whether to share the embeddings",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--neptune_on", action="store_true", help="Whether to upload results to neptune"
    )
    group.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    with open(args.path_opt, "r") as handle:
        options = yaml.load(handle)

    if args.batch_size is not None:
        options["optim"]["batch_size"] = args.batch_size

    print("## args")
    pprint(vars(args))
    print("## options")
    pprint(options)

    if args.help_opt:
        return

    if args.neptune_on:
        neptune_exp = construct_neptune_experiment(args)
        neptune_exp.log_artifact(args.path_opt)
    else:
        neptune_exp = None

    test_loader = load_testdata(options, args)
    vocab_words = test_loader.dataset.vocab_words()
    vocab_answers = test_loader.dataset.vocab_answers()

    model = setup_model(args.share_embeddings, vocab_words, vocab_answers, options)

    exp_logger = construct_logger(options["logs"]["dir_logs"])

    with torch.no_grad():
        evaluate_result = engine.evaluate(
            test_loader, model, exp_logger, neptune_exp=neptune_exp
        )
    if not args.dryrun:
        save_results(evaluate_result, options["logs"]["dir_logs"])

    if args.neptune_on:
        res_file = os.path.join(
            options["logs"]["dir_logs"], "evaluate", "vqa_SemArt_Prediction.json"
        )
        neptune_exp.log_artifact(res_file)
        neptune_exp.stop()
    return


def construct_neptune_experiment(args: Namespace) -> neptune.experiments.Experiment:
    neptune.init("artQA/artQA-IJICAI-submission")
    neptune_exp = neptune.create_experiment(
        name=f"eval-iQAN",
        params=vars(args),
        properties={
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "wd": os.getcwd(),
            "cmd": " ".join(sys.argv),
        },
        tags=["question-image", "iQAN", "evaluation"],
        upload_stdout=True,
    )

    return neptune_exp


def load_testdata(options: dict, args: Namespace) -> torch.utils.data.DataLoader:
    image_dataset_name: str
    if "coco" in options:
        image_dataset_name = "coco"
    elif "clevr" in options:
        image_dataset_name = "clevr"
    else:
        raise RuntimeError("valid dataset is [coco, clevr]")
    image_dataset_opt: dict = options[image_dataset_name]

    testset = datasets.factory_VQA(
        args.dataset_path,
        "test",
        options["vqa"],
        image_dataset_name,
        image_dataset_opt,
        opt_vgenome=options.get("vgnome", None),
    )

    test_loader = testset.data_loader(
        batch_size=options["optim"]["batch_size"], num_workers=1
    )
    return test_loader


def setup_model(
    share_embeddings: bool, vocab_words: list, vocab_answers: list, options: dict
):
    model = getattr(models, options["model"]["arch"])(
        options["model"], vocab_words, vocab_answers
    )

    if share_embeddings:
        model.set_share_parameters()

    path_ckpt_model = os.path.join(options["logs"]["dir_logs"], "best_model.pth.tar")
    model = load_model(model, path_ckpt_model)

    #  Begin evaluation and training
    model = nn.DataParallel(model).cuda()
    return model


def load_model(model: nn.Module, path_ckpt_model: str) -> nn.Module:
    model_state = torch.load(path_ckpt_model)
    model.load_state_dict(model_state)
    return model


def save_results(results: List[Dict], dir_logs: str) -> None:
    result_name = "vqa_SemArt_Prediction.json"
    result_path = os.path.join(dir_logs, "evaluate", result_name)
    os.system("mkdir -p " + os.path.dirname(result_path))
    with open(result_path, "w") as handle:
        json.dump(results, handle)


def construct_logger(name: str) -> logger.Experiment:
    exp_logger = logger.Experiment(name)
    exp_logger.add_meters(
        "test",
        {
            "acc1": logger.AvgMeter(),
            "acc5": logger.AvgMeter(),
            "acc10": logger.AvgMeter(),
            "bleu_score": logger.AvgMeter(),
            "batch_time": logger.AvgMeter(),
        },
    )
    return exp_logger


if __name__ == "__main__":
    main()
