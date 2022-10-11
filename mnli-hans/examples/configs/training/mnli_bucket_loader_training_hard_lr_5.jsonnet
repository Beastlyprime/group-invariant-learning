local hans = import '../dataset/hans.jsonnet';
local mnli_hard = import '../dataset/mnli_hard.jsonnet';
local mnli_m_hard = import '../dataset/mnli_m_hard.jsonnet';

{
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 3
    },
    "num_workers": 1
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 3
    },
    "num_workers": 1
  },
  "trainer": {
    "type": "opendebias_gradient_descent",
    "num_epochs": 3,
    "validation_metric": "+accuracy",
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
            mnli_hard, mnli_m_hard
        ],
      },
    ],
  }
}
