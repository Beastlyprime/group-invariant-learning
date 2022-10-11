import logging
from typing import Dict, Optional

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers import DatasetReader

from opendebias.dataset_readers.text_pair_classification import TextPairClassification

logger = logging.getLogger(__name__)


@DatasetReader.register('fever')
class FEVER(TextPairClassification):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset_name,
            line_format = 'json',
            sentence_a_field = "evidence",
            sentence_b_field = "claim",
            label_field = "gold_label",
            instance_id_field = "id",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = False,
        )

@DatasetReader.register('fever-claim-only')
class FEVERClaimOnly(TextPairClassification):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset_name,
            line_format = 'json',
            sentence_a_field = "claim",
            sentence_b_field = "evidence",
            label_field = "gold_label",
            instance_id_field = "id",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = True,
        )

@DatasetReader.register('fever-symmetric-v1')
class FEVERSymmetricV1(TextPairClassification):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset_name,
            line_format = 'json',
            sentence_a_field = "evidence_sentence",
            sentence_b_field = "claim",
            label_field = "label",
            instance_id_field = "id",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = False,
        )

@DatasetReader.register('fever-symmetric-v2')
class FEVERSymmetricV2(TextPairClassification):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset_name,
            line_format = 'json',
            sentence_a_field = "evidence",
            sentence_b_field = "claim",
            label_field = "gold_label",
            instance_id_field = "id",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = False,
        )


