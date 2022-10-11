"""
The `train` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import json
import logging
import os
import sys
from os import PathLike
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Lazy, Params, Registrable
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.params import parse_overrides, with_fallback
from allennlp.common.plugins import import_plugins
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.models.archival import CONFIG_NAME, archive_model, load_archive
from allennlp.models.model import _DEFAULT_WEIGHTS, Model
from allennlp.training import util as training_util
from allennlp.training.trainer import Trainer
from overrides import overrides

logger = logging.getLogger(__name__)


def read_list_with_comma(string):
    return [ch for ch in string.strip().split(",")]

@Subcommand.register("multi_environment_train")
class MultiEnvironmentTrain(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Multiple Environment Train"""
        subparser = parser.add_parser(self.name, description=description, help="Multi Environment Training")

        subparser.add_argument("--train-set-param-path", 
                               type=str, 
                               required=True,
                               help="path to parameter file describing the train dataset reader and data path")
        subparser.add_argument("--validation-set-param-path", 
                               type=str, 
                               default=None,
                               help="path to parameter file describing the validation dataset reader and data path")
        subparser.add_argument("--model-param-path", 
                                type=str, 
                                required=True,
                                help="path to parameter file describing the main model")
        subparser.add_argument("--training-param-path", 
                                type=str, 
                                required=True,
                                help="path to parameter file describing the trainer")

        subparser.add_argument("--multi-env-loss", type=str, required=True, help="including IRMv1, etc.")
        subparser.add_argument("--loss-args", type=json.loads, default=None)
        subparser.add_argument("--metrics", type=json.loads, default=None)

        subparser.add_argument("--sampler", type=str, default=None, required=True)
        subparser.add_argument("--sampler-args", type=json.loads, default=None, required=False)
        subparser.add_argument("--predefined-group-file", 
                        type=json.loads, 
                        default=None,
                        help="""a json structure describing predefined group file""")
        subparser.add_argument("--group-weights-file", 
                        type=str, 
                        # required=True,
                        default=None,
                        help="""a string describing group weights file path""")

        subparser.add_argument("--seed", type=int)
        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )
    
        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    train_model_from_file(
        trainset_param_path=args.train_set_param_path,
        validset_param_path=args.validation_set_param_path,
        model_param_path=args.model_param_path,
        training_param_path=args.training_param_path,
        predefined_group_file=args.predefined_group_file,
        multi_env_loss=args.multi_env_loss,
        loss_args=args.loss_args,
        sampler=args.sampler,
        sampler_args=args.sampler_args,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
        seed = args.seed,
        args = args
    )


def create_sampler_config(sampler, sampler_args):
    sampler_config = {
        "type": sampler,
        "batch_size": 32
    }
    if sampler_args is not None and len(sampler_args) > 0:
        sampler_config.update(sampler_args)
    return sampler_config

            
def create_multi_env_loss_config(multi_env_loss, multi_env_loss_args):
    multi_env_loss_config = {
        "type": multi_env_loss,
    }
    if multi_env_loss_args is not None and len(multi_env_loss_args) > 0:
        multi_env_loss_config.update(multi_env_loss_args)
    return multi_env_loss_config

def train_model_from_file(
    trainset_param_path: Union[str, PathLike],
    validset_param_path: Union[str, PathLike],
    model_param_path: Union[str, PathLike],
    training_param_path: Union[str, PathLike],
    predefined_group_file: Dict[str, Any],
    multi_env_loss: str,
    loss_args: Dict[str, Any],
    sampler: str,
    sampler_args: Dict[str, Any],
    serialization_dir: Union[str, PathLike],
    overrides: Union[str, Dict[str, Any]] = "",
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    seed: int = None,
    args = None
) -> Optional[Model]:

    # Load the experiment config from a file and pass it to `train_model`.
    trainset_params = Params.from_file(trainset_param_path).as_dict(quiet=True)
    validset_params = Params.from_file(validset_param_path).as_dict(quiet=True) if validset_param_path is not None else None

    dataset_params = trainset_params
    dataset_params['train_data_path'] = dataset_params.pop("data_path")
    if validset_params is not None:
        dataset_params['validation_data_path'] = validset_params.pop("data_path")
        dataset_params['validation_dataset_reader'] = validset_params.pop("dataset_reader")
    model_params = Params.from_file(model_param_path).as_dict(quiet=True)
    training_params = Params.from_file(training_param_path).as_dict(quiet=True)
    multi_env_loss_params = create_multi_env_loss_config(multi_env_loss, loss_args)
    multi_env_loss_params["metrics"] = args.metrics
    multi_env_loss_params["model"] = model_params
    if 'group_weights_file' in args:
        multi_env_loss_params["weights_file_path"] = args.group_weights_file
    model_params = {"model": multi_env_loss_params}

    sampler_args = {} if sampler_args is None else sampler_args
    sampler_args["predefined_group_file"] = predefined_group_file
    sampler_config = {"data_loader": {"batch_sampler": create_sampler_config(sampler, sampler_args)}}
    params = {}
    if seed is not None:
        params['random_seed'] = seed * 10 + 1
        params['numpy_seed'] = seed 
        params['pytorch_seed'] = seed // 10 + 1
    params = with_fallback(preferred=dataset_params, fallback=params)
    params = with_fallback(preferred=training_params, fallback=params)
    params = with_fallback(preferred=model_params, fallback=params)
    params = with_fallback(preferred=sampler_config, fallback=params)

    # process overrides
    if isinstance(overrides, dict):
        overrides = json.dumps(overrides)
    overrides_dict = parse_overrides(overrides)
    params = Params(with_fallback(preferred=overrides_dict, fallback=params))
    return train_model(
        params=params,
        serialization_dir=serialization_dir,
        recover=recover,
        force=force,
        node_rank=node_rank,
        include_package=include_package,
        dry_run=dry_run,
        file_friendly_logging=file_friendly_logging,
        args=args
    )


def train_model(
    params: Params,
    serialization_dir: Union[str, PathLike],
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
    args = None,
) -> Optional[Model]:
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        model = _train_worker(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            include_package=include_package,
            dry_run=dry_run,
            file_friendly_logging=file_friendly_logging,
            args=args
        )

        if not dry_run:
            archive_model(serialization_dir)
        return model

    # Otherwise, we are running multiple processes for training.
    else:
        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1
        num_nodes = distributed_params.pop("num_nodes", 1)

        if not (multi_device or num_nodes > 1):
            raise ConfigurationError(
                "Multiple cuda devices/nodes need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        master_addr = distributed_params.pop("master_address", "127.0.0.1")
        master_port = distributed_params.pop("master_port", 29500)
        num_procs = len(device_ids)
        world_size = num_nodes * num_procs

        # Creating `Vocabulary` objects from workers could be problematic since
        # the data loaders in each worker will yield only `rank` specific
        # instances. Hence it is safe to construct the vocabulary and write it
        # to disk before initializing the distributed context. The workers will
        # load the vocabulary from the path specified.
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(
                params.duplicate(), serialization_dir, print_statistics=dry_run
            )
        params["vocabulary"] = {
            "type": "from_files",
            "directory": vocab_dir,
            "padding_token": vocab._padding_token,
            "oov_token": vocab._oov_token,
        }

        logging.info(
            "Switching to distributed training mode since multiple GPUs are configured | "
            f"Master is at: {master_addr}:{master_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

        mp.spawn(
            _train_worker,
            args=(
                params.duplicate(),
                serialization_dir,
                include_package,
                dry_run,
                node_rank,
                master_addr,
                master_port,
                world_size,
                device_ids,
                file_friendly_logging,
                args
            ),
            nprocs=num_procs,
        )
        if dry_run:
            return None
        else:
            archive_model(serialization_dir)
            model = Model.load(params, serialization_dir)
            return model


def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: List[str] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: List[int] = None,
    file_friendly_logging: bool = False,
    args = None,
) -> Optional[Model]:
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    common_logging.prepare_global_logging(
        serialization_dir,
        rank=process_rank,
        world_size=world_size,
    )
    common_util.prepare_environment(params)

    distributed = world_size > 1

    # not using `allennlp.common.util.is_master` as the process group is yet to be initialized
    master = process_rank == 0

    include_package = include_package or []

    if distributed:
        # Since the worker is spawned and not forked, the extra imports need to be done again.
        # Both the ones from the plugins and the ones from `include_package`.
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # Number of processes per node is useful to know if a process
        # is a master in the local node(node in which it is running)
        os.environ["ALLENNLP_PROCS_PER_NODE"] = str(num_procs_per_node)

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        # Till now, "cuda_device" might not be set in the trainer params.
        # But a worker trainer needs to only know about its specific GPU id.
        params["trainer"]["cuda_device"] = gpu_id
        params["trainer"]["world_size"] = world_size
        params["trainer"]["distributed"] = True

        if gpu_id >= 0:
            torch.cuda.set_device(int(gpu_id))
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=global_rank,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=global_rank,
            )
        logging.info(
            f"Process group of world size {world_size} initialized "
            f"for distributed training in worker {global_rank}"
        )
    train_loop = TrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=process_rank,
    )
    if dry_run:
        return None

    try:
        if distributed:  # let the setup get ready for all the workers
            dist.barrier()

        metrics = train_loop.run()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if master and os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info(
                "Training interrupted by the user. Attempting to create "
                "a model archive using the current best epoch weights."
            )
            archive_model(serialization_dir)
        raise

    if master:
        train_loop.finish(metrics)

    if not distributed:
        return train_loop.model

    return None


class TrainModel(Registrable):
    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        trainer: Trainer,
        evaluation_data_loader: DataLoader = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
    ) -> None:
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.evaluation_data_loader = evaluation_data_loader
        self.evaluate_on_test = evaluate_on_test
        self.batch_weight_key = batch_weight_key

    def run(self) -> Dict[str, Any]:
        return self.trainer.train()

    def finish(self, metrics: Dict[str, Any]):
        if self.evaluation_data_loader is not None and self.evaluate_on_test:
            logger.info("The model will be evaluated using the best epoch weights.")
            test_metrics = training_util.evaluate(
                self.model,
                self.evaluation_data_loader,
                cuda_device=self.trainer.cuda_device,
                batch_weight_key=self.batch_weight_key,
            )

            for key, value in test_metrics.items():
                metrics["test_" + key] = value
        elif self.evaluation_data_loader is not None:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        local_rank: int,
        dataset_reader: DatasetReader,
        train_data_path: str,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        vocabulary: Lazy[Vocabulary] = None,
        datasets_for_vocab_creation: List[str] = None,
        validation_dataset_reader: DatasetReader = None,
        validation_data_path: str = None,
        validation_data_loader: Lazy[DataLoader] = None,
        test_data_path: str = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
    ) -> "TrainModel":
        datasets = training_util.read_all_datasets(
            train_data_path=train_data_path,
            dataset_reader=dataset_reader,
            validation_dataset_reader=validation_dataset_reader,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
        )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in datasets:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

            logger.info(
                "From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation),
            )

        instance_generator = (
            instance
            for key, dataset in datasets.items()
            if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation
            for instance in dataset
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)
        if not vocabulary_:
            vocabulary_ = Vocabulary.from_instances(instance_generator)
        model_ = model.construct(vocab=vocabulary_)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the master. In the degenerate non-distributed
        # case, we're trivially the master. In the distributed case this is safe
        # to do without worrying about race conditions since saving and loading
        # the vocab involves acquiring a file lock.
        if common_util.is_master():
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        for dataset in datasets.values():
            dataset.index_with(model_.vocab)

        data_loader_ = data_loader.construct(dataset=datasets["train"])
        validation_data = datasets.get("validation")
        if validation_data is not None:
            # Because of the way Lazy[T] works, we can't check it's existence
            # _before_ we've tried to construct it. It returns None if it is not
            # present, so we try to construct it first, and then afterward back off
            # to the data_loader configuration used for training if it returns None.
            validation_data_loader_ = validation_data_loader.construct(dataset=validation_data)
            if validation_data_loader_ is None:
                validation_data_loader_ = data_loader.construct(dataset=validation_data)
        else:
            validation_data_loader_ = None

        test_data = datasets.get("test")
        if test_data is not None:
            test_data_loader = validation_data_loader.construct(dataset=test_data)
            if test_data_loader is None:
                test_data_loader = data_loader.construct(dataset=test_data)
        else:
            test_data_loader = None
        
        # We don't need to pass serialization_dir and local_rank here, because they will have been
        # passed through the trainer by from_params already, because they were keyword arguments to
        # construct this class in the first place.
        trainer_ = trainer.construct(
            model=model_,
            data_loader=data_loader_,
            validation_data_loader=validation_data_loader_,
        )

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=test_data_loader,
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )


TrainModel.register("default", constructor="from_partial_objects")(TrainModel)