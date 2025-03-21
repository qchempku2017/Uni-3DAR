#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import sys
import time
import os
import re
from typing import Optional, Any, Callable

import numpy as np
import torch
from unicore import (
    options,
    tasks,
    utils,
)
from unicore.distributed import utils as distributed_utils


def check_files_count(filename, total_count=10000, data_type="molecule"):
    try:
        if data_type == "molecule":
            molecule_count = 0
            with open(filename, "r") as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    try:
                        atom_count = int(line.strip())
                        molecule_count += 1
                        for _ in range(atom_count + 1):
                            file.readline()
                    except ValueError:
                        continue
            if molecule_count == total_count:
                return True
        else:

            with open(filename, "r") as f:
                content = f.read()
            blocks = re.split(r"(?=^[ \t]*data_)", content, flags=re.MULTILINE)
            if "data_" not in blocks[0]:
                blocks = blocks[1:]
            if len(blocks) == total_count:
                return True
    except Exception as e:
        print(e)

    return False


def write_in_file(f, cur_res):
    f.write(cur_res)
    f.write("\n")


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unicore_cli.train")


def main(args) -> None:
    utils.import_user_module(args)
    utils.set_jit_fusion_options()

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.model = "uni3dar_sampler"
    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    assert args.loss, "Please specify loss to train a model"

    # Build model and loss
    model = task.build_model(args)
    state = torch.load(args.finetune_from_model, map_location="cpu", weights_only=False)
    errors = model.load_state_dict(state["ema"]["params"], strict=True)
    print("loaded from {}, errors: {}".format(args.finetune_from_model, errors))

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )

    total_n = args.num_samples
    output_file = args.save_path
    if check_files_count(
        output_file, total_count=total_n, data_type=task.args.data_type
    ):
        return
    try:
        os.system(f"rm {output_file}")
    except:
        pass
    model = model.cuda().bfloat16()
    start = time.time()
    count = 0
    while count < total_n:
        res, _ = model.generate()
        if len(res) > 0:
            with open(f"{output_file}", "a+") as f:
                for cur_res in res:
                    if count % 1000 == 0:
                        print(f"count: {count}")
                    if count < total_n:
                        write_in_file(f, cur_res)
                    count += 1
    end = time.time()
    print(f"Total time: {end - start}")


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    try:
        if args.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(args, main)
        else:
            distributed_utils.call_main(args, main)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            time.sleep(1)
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    cli_main()
