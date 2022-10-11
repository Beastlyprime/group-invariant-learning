{
  "dataset_reader": {
    "type": "hans",
    "dataset_name": "hans",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "pretrained/bert-base-uncased",
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "pretrained/bert-base-uncased",
      }
    }
  },
  "data_path": "data/hans/heuristics_evaluation_set.txt",
  "label_vocab_extras": ["non-entailment"],
}