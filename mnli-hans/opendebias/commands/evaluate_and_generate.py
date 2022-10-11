"""
The `evaluate` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
"""

import argparse
import json
import logging
from typing import Any, Dict

from overrides import overrides
from torch.utils.data import dataset

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.common.checks import check_for_gpu
from torch.cuda import amp
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm
import torch
import os
import pathlib
import numpy as np
# from src.utils.global_config import CONFIG_POOL

logger = logging.getLogger(__name__)


@Subcommand.register("debiased_evaluate")
class DebiasedEvaluate(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate the specified model + dataset"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate the specified model + dataset."
        )

        subparser.add_argument("--archive-file", type=str, help="path to an archived trained model")


        subparser.add_argument(
            "--evaluate-file-config-files", type=json.loads, help="path to the config files describing the evaluation data"
        )

        subparser.add_argument("--bias-only", action='store_true')

        subparser.add_argument(
            "--output-folder", type=str, help="optional path to write the metrics to as JSON"
        )

        subparser.add_argument(
            "--amp", action='store_true'
        )

        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        # subparser.add_argument(
        #     '--save-last-hidden-path', type=str, help='optional path to save last hidden as a file',
        #     default=None
        # )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
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
            "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--val-weights-file",
            type=str,
            default=None,
            help="The path of .json file of val weights.",
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser

# def set_global_config(args):
#     if args.save_last_hidden_path is not None:
#         CONFIG_POOL['save_last_hidden'] = True


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # set_global_config(args)

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    metrics = {}
    # Load all input configs
    for eidx, eval_input_config_path in enumerate(args.evaluate_file_config_files.values()):
        eval_input_config = Params.from_file(eval_input_config_path)
        metrics = evaluate_on_dataset_config(model, eval_input_config, args, metrics, eidx)
    
    # Dump Metrics
    with open(pathlib.Path(args.output_folder) / 'eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    from pprint import pprint 
    pprint(metrics)

    logger.info("Finished evaluating.")
    return metrics


def evaluate_on_dataset_config(model, eval_input_config, args, metrics, eidx):
    dataset_reader = DatasetReader.from_params(eval_input_config.pop("dataset_reader"))
    data_path = eval_input_config.pop("data_path")
    instances = dataset_reader.read(data_path)
    if "label_vocab_extras" in eval_input_config:
        model.vocab.add_tokens_to_namespace(eval_input_config.pop("label_vocab_extras"), "labels")
    instances.index_with(model.vocab)
    data_loader = PyTorchDataLoader(
        dataset = instances,
        num_workers = 4,
        shuffle = False,
        batch_size = 64 if args.batch_size is None else args.batch_size
    )
    iterator = iter(data_loader)
    generator_tqdm = Tqdm.tqdm(iterator, desc=f"Evaluate on {dataset_reader._dataset_name}", total=len(data_loader))
    check_for_gpu(args.cuda_device)

    with torch.no_grad():
        if args.bias_only:
            # bias only output log logits
            metrics, ret = eval_on_dataset_bias_only(
                model, dataset_reader._dataset_name, dataset_reader, generator_tqdm, metrics, args, eidx)
        else:
            metrics, ret = eval_on_dataset(
                model, dataset_reader._dataset_name, dataset_reader, generator_tqdm, metrics, args, eidx)

    # save predictions
    predictions_output_folder = args.output_folder
    if not os.path.exists(predictions_output_folder):
        os.makedirs(predictions_output_folder)
    with open(pathlib.Path(predictions_output_folder) / f'{dataset_reader._dataset_name}.json', 'w') as f:
        json.dump(ret, f)
    return metrics


def eval_on_dataset_bias_only(model, dataset_name, dataset_reader, data_generator, metrics, args, eidx):
    model.eval()

    all_logits = []
    all_probs = []
    all_labels = []
    all_ids = []
    all_metas = []
    all_indexs = []
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    
    for batch in data_generator:
        with amp.autocast(args.amp):
            batch = nn_util.move_to_device(batch, args.cuda_device)
            all_labels.append(batch.pop('label'))
            all_indexs.append(batch.pop('index'))
            all_ids.extend([b['instance_id'] for b in batch['metadata']])
            batch_outputs = model(**batch)
            all_logits.append(batch_outputs['logits'])
            all_probs.append(batch_outputs['probs'])
        
    y = torch.cat(all_labels)
    all_indexs = torch.cat(all_indexs)
    ret_probs = {}
    probs = torch.cat(all_probs)
    logits = torch.cat(all_logits)
    cur_metrics = dataset_reader.eval(model.vocab, probs, y, all_metas)
    for k, v in cur_metrics.items():
        metrics[f'{dataset_name}-{k}'] = v 

    ret_probs['id'] = all_ids
    # ret_probs['log_probs'] = log_softmax(logits).detach().cpu().numpy().tolist() # return log softmax
    ret_probs['logits'] = logits.detach().cpu().numpy().tolist() # return log softmax
    ret_probs['y'] = y.detach().cpu().numpy().tolist()
    ret_probs['index2token'] = model.vocab.get_index_to_token_vocabulary("labels")
    ret_probs['index'] = all_indexs.detach().cpu().numpy().tolist()
    return metrics, ret_probs


def eval_on_dataset(model, dataset_name, dataset_reader, data_generator, metrics, args, eidx):
    model.eval()
    all_main_logits = []
    all_bias_logits = []
    all_ensemble_logits = []
    all_labels = []
    all_ids = []
    all_metas = []
    all_last_hiddens = []
    all_indexs = []
    all_weights = []
    for batch in data_generator:
        with amp.autocast(args.amp):
            batch = nn_util.move_to_device(batch, args.cuda_device)
            all_labels.append(batch.pop('label'))
            all_indexs.append(batch['index'])
            all_ids.extend([b['instance_id']  for b in batch['metadata']])
            batch_outputs = model(**batch)
            if 'logits' in batch_outputs:
                all_main_logits.append(batch_outputs['logits'])
            elif 'main_logits' in batch_outputs:
                all_main_logits.append(batch_outputs['main_logits'])
            if 'bias_only_logits' in batch_outputs:
                all_bias_logits.append(batch_outputs['bias_only_logits'])
            if 'ensemble_logits' in batch_outputs:
                all_ensemble_logits.append(batch_outputs['ensemble_logits'])
            if 'metadata' in batch: all_metas.extend(batch['metadata'])

    if args.val_weights_file is not None and ('mnli' in dataset_name):
        all_weights = np.array([])
        val_weights = json.load(open(args.val_weights_file))
        for index in all_indexs:
            batch_index = np.array(index.cpu())
            all_weights = np.append(all_weights, batch_index)
    y = torch.cat(all_labels)
    ret  = {}
    for split, logits in zip(('main', 'bias_only', 'ensemble'), (all_main_logits, all_bias_logits, all_ensemble_logits)):
        if len(logits):
            logits = torch.cat(logits)
            if len(all_weights) > 0:
                cur_metrics = dataset_reader.eval(model.vocab, logits, y, all_metas, all_weights)
                ad_metric = dataset_reader.eval(model.vocab, logits, y, all_metas)
                for k, v in ad_metric.items():
                    metrics[f'mnli_val-{k}'] = v
            else:
                cur_metrics = dataset_reader.eval(model.vocab, logits, y, all_metas)
            for k, v in cur_metrics.items():
                metrics[f'{dataset_name}-{split}-{k}'] = v 
            ret[split] = logits.detach().cpu().numpy().tolist()
    ret['instance_id'] = [str(id_) for id_ in all_ids]
    ret['y'] = y.detach().cpu().numpy().tolist()
    # dump vocab
    ret['index2token'] = model.vocab.get_index_to_token_vocabulary("labels")

    # from pathlib import Path
    # save last hidens
    # if CONFIG_POOL['save_last_hidden']:
        # all_last_hiddens = torch.cat(all_last_hiddens)
        # if not os.path.exists(args.save_last_hidden_path):
            # os.makedirs(args.save_last_hidden_path, exist_ok=True)
        # torch.save(all_last_hiddens, Path(args.save_last_hidden_path)/"{}.pt".format(dataset_name))
    return metrics, ret