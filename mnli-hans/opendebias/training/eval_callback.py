from allennlp.training.trainer import EpochCallback
from typing import List, Tuple, Dict, Any
from allennlp.data import DatasetReader, Vocabulary
import torch
import logging
from allennlp.training import util as training_util
from allennlp.common import util
import torch.distributed as dist
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler, BatchSampler
from allennlp.common import Params
from torch.cuda import amp
import os

logger = logging.getLogger(__name__)

@EpochCallback.register("eval_epoch_callback")
class EvalEpochCallback:
    def __init__(self, eval_datasets: Params, bias_only: bool = False):
        params = eval_datasets
        self.eval_readers = [DatasetReader.from_params(param.pop("dataset_reader")) for param in params]
        self.dataset_names = [reader._dataset_name for reader in self.eval_readers]

        self.eval_datasets = []
        for reader, param in zip(self.eval_readers, params):
            self.eval_datasets.append(reader.read(param.pop("data_path")))
        self.batch_samplers = None
        self.params = params

        self.has_indexed = False
        self.label_vocab_extras = set()
        for param in params:
            if "label_vocab_extras" in param:
                for label in param.pop("label_vocab_extras"):
                    self.label_vocab_extras.add(label)
        self.label_vocab_extras = list(self.label_vocab_extras)
        self.bias_only = bias_only

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        self.trainer = trainer
        if not self.has_indexed:
            vocabulary = trainer.model.vocab
            if self.label_vocab_extras:
                vocabulary.add_tokens_to_namespace(self.label_vocab_extras, "labels")
            for dataset in self.eval_datasets:
                dataset.index_with(vocabulary)

            self.batch_samplers = []
            for idx, param in enumerate(self.params):
                if 'batch_sampler' in param:
                    p = param.pop("batch_sampler") 
                    p['data_source'] = self.eval_datasets[idx]
                    self.batch_samplers.append(BatchSampler.from_params(p))
                else:
                    self.batch_samplers.append(None)
            self.has_indexed = True 
        
        if epoch == -1:
            return
        
        metrics = {"epoch": epoch}
        for dataset_name, dataset, reader, batch_sampler in zip(self.dataset_names, self.eval_datasets, self.eval_readers, self.batch_samplers):
            if batch_sampler is not None:
                data_generator = PyTorchDataLoader(
                    dataset = dataset,
                    num_workers = 1,
                    batch_sampler = batch_sampler,
                )
            else:
                data_generator = PyTorchDataLoader(
                    dataset = dataset,
                    num_workers = 1,
                    shuffle = False,
                    batch_size = 64
                )
            with torch.no_grad():
                metrics.update(self.eval_on_dataset(dataset_name, reader, data_generator, metrics))
                # It is safe again to wait till the validation is done. This is
                # important to get the metrics right.
                if self.trainer._distributed:
                    dist.barrier()
            logger.info('Evaluate done on {}'.format(dataset_name))
        util.dump_metrics(os.path.join(self.trainer._serialization_dir, f"test_metrics_epoch_{epoch}.json"), metrics)
        metrics.pop("epoch")
        self.trainer._tensorboard.log_metrics(train_metrics={}, val_metrics=metrics, log_to_console=True, epoch=epoch + 1)

        

    def eval_on_dataset(self, dataset_name, reader, data_generator, metrics):
        self.trainer._pytorch_model.eval()
        all_main_probs = []
        all_bias_probs = []
        all_ensemble_probs = []
        all_labels = []
        all_metas = []
        all_accs = []
        for batch in data_generator:
            if self.trainer._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing validation early! "
                        "This implies that there is an imbalance in your validation "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break
            with amp.autocast(self.trainer._use_amp):
                all_labels.append(batch.pop('label')) # for hans
                # all_labels.append(batch['label'])
                batch_outputs = self.trainer.batch_outputs(batch, for_training=False)
                if 'probs' in batch_outputs or 'main_probs' in batch_outputs:
                    key = 'main_probs' if 'main_probs' in batch_outputs else 'probs'
                    all_main_probs.append(batch_outputs[key])
                    if self.bias_only: continue
                elif 'bias_only_probs' in batch_outputs:
                    all_bias_probs.append(batch_outputs['bias_only_probs'])
                elif 'ensemble_probs' in batch_outputs:
                    all_ensemble_probs.append(batch_outputs['ensemble_probs'])
                if 'metadata' in batch: all_metas.extend(batch['metadata'])
        ret_metrics = self.trainer._pytorch_model.get_metrics(reset=True)
        y = torch.cat(all_labels)
        # print(torch.cat(all_main_probs))
        if self.bias_only:
            probs = torch.cat(all_main_probs)
            cur_metrics = reader.eval(self.trainer.model.vocab, probs, y, all_metas)
            for k, v in cur_metrics.items():
                metrics[f'{dataset_name}-{k}'] = v 
        else:
            for split, probs in zip(('main', 'bias_only', 'ensemble'), (all_main_probs, all_bias_probs, all_ensemble_probs)):
                if len(probs):
                    probs = torch.cat(probs)
                    cur_metrics = reader.eval(self.trainer.model.vocab, probs, y, all_metas)
                    for k, v in cur_metrics.items():
                        metrics[f'{dataset_name}-{split}-{k}'] = v 
        for key, val in ret_metrics.items():
            metrics[f'{dataset_name}-{key}'] = val
        return metrics
        

            
            





