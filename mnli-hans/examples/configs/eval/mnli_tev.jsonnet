{
  "dataset_reader": {
    "type": "mnli",
    "dataset_name": "mnli_tev",
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
  "data_path": "data/glue_data/MNLI/dev_matched.tsv",
}