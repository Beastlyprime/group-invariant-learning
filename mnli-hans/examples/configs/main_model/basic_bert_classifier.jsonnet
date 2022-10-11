{
    "type": "opendebias_basic_classifier",
    "dropout": 0.1,
    "feedforward": {
        "activations": "tanh",
        "hidden_dims": 768,
        "input_dim": 768,
        "num_layers": 1
    },
    "initializer": {
        "regexes": [
            [
                ".*_feedforward._linear_layers.*.weight",
                {
                    "mean": 0,
                    "std": 0.02,
                    "type": "normal"
                }
            ],
            [
                ".*_classification_layer.weight.*",
                {
                    "mean": 0,
                    "std": 0.02,
                    "type": "normal"
                }
            ],
            [
                ".*_feedforward._linear_layers.*.bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*_classification_layer.bias.*",
                {
                    "type": "zero"
                }
            ]
        ]
    },
    "seq2vec_encoder": {
        "type": "cls_pooler",
        "embedding_dim": 768
    },
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": "pretrained/bert-base-uncased"
            }
        }
    }
}