import argparse
from typing import Optional, Sequence

from hydra import initialize_config_module

from sam2.training.train import main as _main
from sam2.training.utils.train_utils import register_omegaconf_resolvers


def cli(argv: Optional[Sequence[str]] = None) -> None:
    initialize_config_module("sam2", version_base="1.2")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--use-cluster", type=int, default=None)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    args = parser.parse_args(args=None if argv is None else list(argv))
    register_omegaconf_resolvers()
    _main(args)


def run(config: str, **kwargs) -> None:
    initialize_config_module("sam2", version_base="1.2")
    class Args:
        def __init__(self, **kw):
            self.config = config
            self.use_cluster = kw.get("use_cluster", None)
            self.partition = kw.get("partition", None)
            self.account = kw.get("account", None)
            self.qos = kw.get("qos", None)
            self.num_gpus = kw.get("num_gpus", None)
            self.num_nodes = kw.get("num_nodes", None)
    register_omegaconf_resolvers()
    _main(Args(**kwargs))
