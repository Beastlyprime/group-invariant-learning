import logging
from typing import Dict, Optional
from overrides import overrides
import torch
import numpy as np
from allennlp.data.vocabulary import Vocabulary

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers import DatasetReader

from opendebias.dataset_readers.text_pair_classification import TextPairClassification

logger = logging.getLogger(__name__)


@DatasetReader.register('mnli')
class MNLI(TextPairClassification):
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
            sentence_a_field = "sentence1",
            sentence_b_field = "sentence2",
            label_field = "gold_label",
            instance_id_field = "index",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = False,
        )
    @overrides
    def eval(self, vocabulary: Vocabulary, probabilities: torch.Tensor, 
                   labels: torch.Tensor, metadatas: Optional[Dict[str, object]] = None, weights: Optional[np.array] = None):
        _, predicts = torch.max(probabilities, dim=-1)
        predicts = predicts.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        if weights is not None:
            accuracy = sum((predicts == labels) * weights) / sum(weights)
        else:
            accuracy = sum(predicts == labels) / len(predicts)
        return {
            'accuracy': accuracy
        }


@DatasetReader.register('mnli-hypothesis-only')
class MNLIHypothesisOnly(TextPairClassification):
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
            sentence_a_field = "sentence2",
            sentence_b_field = "sentence1",
            label_field = "gold_label",
            instance_id_field = "index",
            index_field = None,
            skip_index_label = False,
            tokenizer = tokenizer,
            token_indexers = token_indexers,
            combine_input_fields = combine_input_fields,
            partial_input = True,
        )


@DatasetReader.register('hans')
class HANS(TextPairClassification):
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
            sentence_a_field = "sentence1",
            sentence_b_field = "sentence2",
            label_field = "gold_label",
            instance_id_field = "pairID",
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

        entailment_id = vocabulary.get_token_index("entailment", namespace="labels")
        non_entailment_id = vocabulary.get_token_index("non-entailment", namespace="labels")
        contradiction_id = vocabulary.get_token_index("contradiction", namespace="labels")
        entailment_id = vocabulary.get_token_index("entailment", namespace="labels")
        neutral_id = vocabulary.get_token_index("neutral", namespace="labels")

        probs = probabilities
        y = labels

        # set non-entailment is 0 and entailment is 1
        y_2class = torch.where(labels == entailment_id, torch.ones_like(y), torch.zeros_like(y)).to(probs.device)
        y = y.cpu().numpy()
        entail_mask = (y == entailment_id)
        non_entail_mask = (y == non_entailment_id)
        
        # max scheme
        _, predict = torch.max(probs, dim=-1)
        predict = predict.detach().cpu().numpy()
        predict = np.where(predict != entailment_id, non_entailment_id, entailment_id)
        max_accuracy = sum(predict == y) / len(predict)

        max_entail_accuracy = sum(predict[entail_mask] == y[entail_mask]) / sum(entail_mask)
        max_non_entail_accuracy = sum(predict[non_entail_mask] == y[non_entail_mask]) / sum(non_entail_mask)

        max_scheme_2class_probs = torch.zeros(len(probs), 2).to(probs.device)
        # set non-entailment is 0 and entailment is 1
        max_scheme_2class_probs[:, 0] = torch.where(probs[:, contradiction_id] > probs[:, neutral_id], 
                                                    probs[:, contradiction_id], probs[:, neutral_id])
        max_scheme_2class_probs[:, 1] = probs[:, entailment_id]
        assert abs((torch.sum(torch.max(max_scheme_2class_probs, dim=-1)[1] == y_2class).detach().cpu().item() / len(y_2class)) - max_accuracy) < 0.001

        # sum scheme 
        probs_torch = probs.detach()
        probs = probs.detach().cpu().numpy()
        probs_4class = np.zeros((len(probs), 4))
        probs_4class[:, entailment_id] = probs[:, entailment_id]
        for idx in range(4):
            if idx != entailment_id and idx != non_entailment_id:
                probs_4class[:, non_entailment_id] += probs[:, idx]
                probs_4class[:, idx] -= 1000
        predict = np.argmax(probs_4class, axis=1)
        sum_accuracy = sum(predict == y) / len(predict)
        sum_entail_accuracy = sum(predict[entail_mask] == y[entail_mask]) / sum(entail_mask)
        sum_non_entail_accuracy = sum(predict[non_entail_mask] == y[non_entail_mask]) / sum(non_entail_mask)

        sum_scheme_2class_probs = torch.zeros(len(probs_torch), 2).to(probs_torch.device)
        sum_scheme_2class_probs[:, 0] = probs_torch[:, contradiction_id] + probs_torch[:, neutral_id]
        sum_scheme_2class_probs[:, 1] = probs_torch[:, entailment_id]
        assert abs((torch.sum(torch.max(sum_scheme_2class_probs, dim=-1)[1] == y_2class).detach().cpu().item() / len(y_2class)) - sum_accuracy) < 0.001
        return {
            'accuracy-max': max_accuracy,
            'entail-accuracy-max': max_entail_accuracy,
            'non-entail-accuracy-max': max_non_entail_accuracy,
            'accuracy-sum': sum_accuracy,
            'entail-accuracy-sum': sum_entail_accuracy,
            'non-entail-accuracy-sum': sum_non_entail_accuracy
        }