import random
import yaml
import argparse
import numpy as np
import torch
import builder
import tempfile


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def change_value(config_dict, args):
    def _find_and_replace(obj, key, val):
        if key in obj:
            if val is not None:
                obj[key] = val
            return None
        for k, v in obj.items():
            if isinstance(v, dict):
                return _find_and_replace(v, key, val)  # added return statement

    args = vars(args)
    for key in args.keys():
        for conf_k in config_dict.keys():
            _find_and_replace(config_dict[conf_k], key, args[key])
    return config_dict


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
        default=1000,
        help="max number of generation iteration.",
    )
    parser.add_argument(
        "--eval-ep-num",
        type=int,
        default=5,
        help="number of model evaluaion per iteration.",
    )
    parser.add_argument("--log", action="store_false", help="wandb log")
    parser.add_argument(
        "--save-model-period",
        type=int,
        default=10,
        help="save model for every n iteration.",
    )

    parser.add_argument("--init-sigma", type=float)
    parser.add_argument("--sigma-decay", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--elite-num", type=int)
    parser.add_argument("--offspring-num", type=int)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    config = change_value(config, args)

    set_seed(args.seed)

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
