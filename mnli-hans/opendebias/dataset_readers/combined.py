import itertools
import json
import logging
import os
import torch
from typing import Dict, Optional, List

import torch.distributed as dist
from allennlp.common.file_utils import cached_path
from allennlp.common.util import is_distributed
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.data.fields import Field, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import (PretrainedTransformerTokenizer,
                                      SpacyTokenizer, Tokenizer)
from overrides import overrides

READER_DEBUG=int(os.environ.get("READER_DEBUG", 0))

logger = logging.getLogger(__name__)


@DatasetReader.register('combined')
class Combined(DatasetReader):
    def __init__(
        self,
        main_model_dataset_reader: DatasetReader,
        bias_only_model_dataset_reader: DatasetReader,
        drop_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._main_model_dataset_reader = main_model_dataset_reader
        self._bias_only_model_dataset_reader = bias_only_model_dataset_reader
        if drop_fields is None:
            self._drop_fields = set(["index", "metadata"])
        else:
            self._drop_fields = set(drop_fields)

    @overrides
    def read(self, file_path: str):
        main_model_dataset = self._main_model_dataset_reader.read(file_path)
        bias_only_model_dataset = self._bias_only_model_dataset_reader.read(file_path)
        assert len(main_model_dataset) == len(bias_only_model_dataset)
        for mi, bi in zip(main_model_dataset, bias_only_model_dataset):
            assert mi['metadata']["instance_id"] == bi['metadata']["instance_id"]
        # combine main model instance and bias_only model instance
        combined_instances = []
        for mi, bi in zip(main_model_dataset, bias_only_model_dataset):
            for name, val in bi.fields.items():
                if name not in self._drop_fields:
                    mi.add_field("bias_only_{}".format(name), val)
            combined_instances.append(mi)
        return AllennlpDataset(combined_instances)
        
    def eval(self, vocabulary: Vocabulary, probabilities: torch.Tensor, 
                   labels: torch.Tensor, metadatas: Optional[Dict[str, object]] = None):
        output_dict = {}
        for m_name, m_val in self._main_model_dataset_reader.eval(vocabulary, probabilities, labels, metadatas).items():
            output_dict['main-{}'.format(m_name)] = m_val
        for m_name, m_val in self._bias_only_model_dataset_reader.eval(vocabulary, probabilities, labels, metadatas).items():
            output_dict['bias-only-{}'.format(m_name)] = m_val
        return output_dict