{
  "dataset_reader": {
    "type": "hans",
    "dataset_name": "hans_oracle",
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
  "data_path": "data/hans/oracle_set.txt",
  "label_vocab_extras": ["non-entailment"],
}