local transformer_model = "pretrained/bert-base-uncased";
{
  "dataset_reader": {
    "type": "mnli-hypothesis-only",
    "dataset_name": "mnli_train_hypothesis_only",
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
  "data_path": "../data/glue_data/MNLI/train.tsv",
}