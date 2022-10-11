local hans = import '../dataset/hans.jsonnet';

{
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    },
    "num_workers": 1
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 64
    },
    "num_workers": 1
  },
  "trainer": {
    "type": "opendebias_group_based",
    "num_epochs": 3,
    "validation_metric": "+accuracy",
    "weight_adapt_ascend_ratio": 0.3,
    "learning_rate_scheduler": {
      "type": "auto_linear_with_warmup_ratio",
      "warmup_ratio": 0.1
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 5e-5,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*LayerNorm.weight.*", ".*.bias"], {"weight_decay": 0.0}]
      ]
    },
    "epoch_callbacks": [
      {"type": "track_epoch_callback"},
      {
        "type": "eval_epoch_callback",
        "eval_datasets": [
            hans
        ],
      },
    ],
  }
}
