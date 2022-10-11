local mnli_train = import "../mnli_train.jsonnet";
local mnli_train_hyp_only = import "../bias-only/mnli_train_hyp_only.jsonnet";

{
    "dataset_reader": {
        "type": "combined",
        "main_model_dataset_reader": mnli_train.dataset_reader,
        "bias_only_model_dataset_reader": mnli_train_hyp_only.dataset_reader, 
    },
    "data_path": mnli_train.data_path
}