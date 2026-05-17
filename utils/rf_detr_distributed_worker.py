"""Worker entrypoint for RF-DETR training.

On Linux: launched by torch.distributed.run (handles process group setup).
On Windows: launched directly as a subprocess (single-process training).
"""

import argparse
import json
import os

from rfdetr import RFDETRNano


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-weights", required=True, type=str)
    parser.add_argument("--params-json", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("ALBUMENTATIONS_DISABLE", "1")

    with open(args.params_json, "r") as f:
        training_params = json.load(f)

    model = RFDETRNano(pretrain_weights=args.pretrain_weights)
    model.train(**training_params)


if __name__ == "__main__":
    main()
