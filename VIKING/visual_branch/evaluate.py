from typing import List, Dict
import os
import yaml
import json
from pprint import pprint

import torch
import torch.nn as nn

import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.datasets as datasets

# task specific package
import models.dual_model as models
import dual_model.lib.engine_v2 as engine

from train import construct_dataloader, construct_neptune_experiment
import click

os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = "true"

model_names: List[str] = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)


@click.command()
@click.argument("log_dir", type=click.Path(exists=True, dir_okay=True))
@click.option(
    "--neptlog/--no-neptlog",
    default=False,
    help="upload the experiment to neptune.ai",
)
def main(log_dir: str, neptlog: bool):
    opt_file = os.path.join(log_dir, "params.yml")
    with open(opt_file, "r") as handle:
        options = yaml.load(handle)

    if neptlog:
        neptune_exp = construct_neptune_experiment(
            options["neptune"]["proj_name"],
            options["neptune"]["exp_name"],
            tags=["evaluation"],
        )
        neptune_exp.log_artifact(opt_file)
    else:
        neptune_exp = None

    test_loader = construct_dataloader("test", options)
    vocab_words = test_loader.dataset.vocab_words()
    vocab_answers = test_loader.dataset.vocab_answers()

    model = setup_model(vocab_words, vocab_answers, options)

    exp_logger = construct_logger(options["logs"]["dir_logs"])

    with torch.no_grad():
        evaluate_result = engine.evaluate(
            test_loader, model, exp_logger, neptune_exp=neptune_exp
        )
    save_results(evaluate_result, options["logs"]["dir_logs"])

    if neptlog:
        res_file = os.path.join(
            options["logs"]["dir_logs"],
            "evaluate",
            "Prediction.json",
        )
        neptune_exp.log_artifact(res_file)
        neptune_exp.stop()
    return


def setup_model(
    vocab_words: list,
    vocab_answers: list,
    options: dict,
):
    model = getattr(models, options["model"]["arch"])(
        options["model"], vocab_words, vocab_answers
    )

    path_ckpt_model = os.path.join(
        options["logs"]["dir_logs"], "best_model.pth.tar"
    )
    model = load_model(model, path_ckpt_model)

    #  Begin evaluation and training
    model = nn.DataParallel(model).cuda()
    return model


def load_model(model: nn.Module, path_ckpt_model: str) -> nn.Module:
    model_state = torch.load(path_ckpt_model)
    model.load_state_dict(model_state)
    return model


def save_results(results: List[Dict], dir_logs: str) -> None:
    result_name = "Prediction.json"
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
