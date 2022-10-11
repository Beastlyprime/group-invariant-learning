local transformer_model = "pretrained/bert-base-uncased";
{
  "dataset_reader": {
    "type": "hans",
    "dataset_name": "hans",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    }
  },
  "data_path": "data/hans/heuristics_evaluation_set.txt",
  "label_vocab_extras": ["non-entailment"],
}