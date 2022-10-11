import logging
import torch
from typing import Dict, Optional
from overrides import overrides

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary

from opendebias.dataset_readers.text_pair_classification import TextPairClassification
from sklearn.metrics import f1_score, roc_curve, auc

logger = logging.getLogger(__name__)

@DatasetReader.register('qqp')
class QQP(TextPairClassification):
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
            line_format = 'tsv',
            sentence_a_field = "question1",
            sentence_b_field = "question2",
            label_field = "label",
            instance_id_field = "id",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = False,
        )

    @overrides
    def eval(self, vocabulary: Vocabulary, probabilities: torch.Tensor, 
                   labels: torch.Tensor, metadatas: Optional[Dict[str, object]] = None):
        _, predicts = torch.max(probabilities, dim=-1)
        predicts = predicts.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = sum(predicts == labels) / len(predicts)
        output_dict = {'accuracy': accuracy}

        # split
        dup_index = (labels == vocab.get_token_index("duplicate", namespace="labels"))
        non_dup_index = (labels == vocab.get_token_index("non-duplicate", namespace="labels"))
        for split, index in (('dup', dup_index), ('non-dup', non_dup_index)):
            split_predicts = predicts[index]
            split_y = labels[index]
            output_dict['{}_accuracy'.format(split)] = sum(split_predicts == split_y) / len(split_predicts)  
        
        f1 = f1_score(labels, predicts, pos_label=1)
        f1_nondup = f1_score(labels, predicts, pos_label=0)
        
        fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=0)
        auc_dup = auc(fpr, tpr)
        fpr, tpr, thresholds = roc_curve(labels, predicts, pos_label=1)
        auc_non_dup = auc(fpr, tpr)
        
        output_dict['dup_f1'] = float(f1)
        output_dict['non-dup_f1'] = float(f1_nondup)
        output_dict['dup_auc'] = float(auc_dup)
        output_dict['non-dup_auc'] = float(auc_non_dup)
        
        return output_dict