import itertools
import json
import logging
import os
import torch
from typing import Dict, Optional

import torch.distributed as dist
from allennlp.common.file_utils import cached_path
from allennlp.common.util import is_distributed
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import (PretrainedTransformerTokenizer,
                                      SpacyTokenizer, Tokenizer)
from overrides import overrides

READER_DEBUG=int(os.environ.get("READER_DEBUG", 0))

logger = logging.getLogger(__name__)


@DatasetReader.register('text_pair_classification')
class TextPairClassification(DatasetReader):
    def __init__(
        self,
        dataset_name: str,
        line_format: str = None,
        sentence_a_field: str = None,
        sentence_b_field: str = None,
        label_field: str = None,
        instance_id_field: str = None,
        index_field: Optional[str] = None,
        skip_index_label: Optional[bool] = False,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        partial_input: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)
        self._dataset_name = dataset_name
        self._line_format = line_format
        self._sentence_a_field = sentence_a_field
        self._sentence_b_field = sentence_b_field
        self._label_field = label_field
        self._instance_id_field = instance_id_field
        self._index_field = index_field
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)
        self._partial_input = partial_input
        self._skip_index_label = skip_index_label

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        if is_distributed():
            start_index = dist.get_rank()
            step_size = dist.get_world_size()
            logger.info(
                "Reading %s instances %% %d from tsv dataset at: %s", 
                self._dataset_name, step_size, file_path
            )
        else:
            start_index = 0
            step_size = 1
            logger.info("Reading %s instances from tsv dataset at: %s", self._dataset_name, file_path)

        with open(file_path, "r") as textpair_file:
            if self._line_format == 'tsv' or self._line_format == 'csv':
                separator = '\t' if self._line_format == 'tsv' else ','
                columns = next(textpair_file).strip().split(separator)
                col2id_mapping = {col:idx for idx, col in enumerate(columns)}
                col2id = lambda x: col2id_mapping[x]
                example_iter = (line.strip().split(separator) for line in textpair_file)
            elif self._line_format == 'json':
                col2id = lambda x:x
                example_iter = (json.loads(line) for line in textpair_file)
            else:
                raise NotImplementedError('Only support tsv, csv, and json format')

            line_id = 0
            end_index = None
            if READER_DEBUG == 1:
                step_size = 1000
            elif READER_DEBUG == 2:
                end_index = 1000

            for example in itertools.islice(example_iter, start_index, end_index, step_size):
                label = example[col2id(self._label_field)]
                sentence_a = example[col2id(self._sentence_a_field)]
                sentence_b = example[col2id(self._sentence_b_field)]
                index = int(example[col2id(self._index_field)]) if self._index_field is not None else line_id
                instance_id = example[col2id(self._instance_id_field)]
                yield self.text_to_instance(sentence_a, sentence_b, label, 
                                            index, instance_id, self._skip_index_label)
                line_id += step_size
    

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence_a: str,
        sentence_b: str,
        label: str = None,
        index: str = None,
        instance_id: str = None,
        skip_index_label: bool = False,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        if self._partial_input: # only use sentence_a as the input
            assert sentence_a is not None
            sentence_a = self._tokenizer.tokenize(sentence_a)
            sentence_a = self._tokenizer.add_special_tokens(sentence_a, tokens2=None)
            fields["tokens"] = TextField(sentence_a, self._token_indexers)
            fields["metadata"] = MetadataField({"dataset_name": self._dataset_name, "instance_id": instance_id})
        else:
            assert sentence_a is not None and sentence_b is not None
            sentence_a = self._tokenizer.tokenize(sentence_a)
            sentence_b = self._tokenizer.tokenize(sentence_b)
            if self._combine_input_fields:
                tokens = self._tokenizer.add_special_tokens(sentence_a, sentence_b)
                fields["tokens"] = TextField(tokens, self._token_indexers)
                fields["metadata"] = MetadataField({"dataset_name": self._dataset_name, "instance_id": instance_id})
            else:
                sentence_a_tokens = self._tokenizer.add_special_tokens(sentence_a)
                sentence_b_tokens = self._tokenizer.add_special_tokens(sentence_b)
                fields["sentence_a"] = TextField(sentence_a_tokens, self._token_indexers)
                fields["sentence_b"] = TextField(sentence_b_tokens, self._token_indexers)
                metadata = {
                    "sentence_a_tokens": [x.text for x in sentence_a],
                    "sentence_b_tokens": [x.text for x in sentence_b],
                    "dataset_name": self._dataset_name,
                    "instance_id": instance_id
                }
                fields["metadata"] = MetadataField(metadata)

        if index is not None:
            fields["index"] = LabelField(index,
                                         label_namespace="*instance_index", 
                                         skip_indexing=True)
        if label is not None:
            fields["label"] = LabelField(label,
                                         skip_indexing=skip_index_label)
        return Instance(fields)

    def eval(self, vocabulary: Vocabulary, probabilities: torch.Tensor, 
                   labels: torch.Tensor, metadatas: Optional[Dict[str, object]] = None):
        _, predicts = torch.max(probabilities, dim=-1)
        predicts = predicts.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = sum(predicts == labels) / len(predicts)
        return {
            'accuracy': accuracy
        }