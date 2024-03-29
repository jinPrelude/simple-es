import random
import yaml
import argparse
import numpy as np
import torch
import builder


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="conf/lunarlander.yaml",
        help="config file to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument(
        "--process-num", type=int, default=12, help="number of mp process."
    )
    parser.add_argument(
        "--generation-num",
        type=int,
        default=10000,
        help="max number of generation iteration.",
    )
    parser.add_argument(
        "--eval-ep-num",
        type=int,
        default=5,
        help="number of model evaluaion per iteration.",
    )
    parser.add_argument("--log", action="store_true", help="wandb log")
    parser.add_argument(
        "--save-model-period",
        type=int,
        default=10,
        help="save model for every n iteration.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    loop = builder.build_loop(
        config,
        args.generation_num,
        args.process_num,
        args.eval_ep_num,
        args.log,
        args.save_model_period,
    )
    loop.run()


if __name__ == "__main__":
    main()
