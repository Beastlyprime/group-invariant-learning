import torch
from allennlp.models import Model
from typing import List, Dict, Optional
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive



class LoadableModel(Model):
    def __init__(self, 
                 vocab: Vocabulary, 
                 load_weight_configs: Optional[List[Dict[str, str]]] = None,
                 **kwargs) -> None:
        super().__init__(vocab, **kwargs)
        self._load_weight_configs = load_weight_configs # [{"archive_file_path":xx, "weight_file_path": xx, "to_module_path": "xx,xx", "from_module_path": "xx,xx", "fix": "xx,xx"}]
    
    def _load_weight_file(self, archive_file_path: str, weights_file_path: str = None):
        model = load_archive(archive_file_path, 
                             cuda_device=-1, 
                             weights_file=weights_file_path).model 
        return model


    def _loacate_and_replace(self, from_path, to_path, from_model, fix):
        to_model = self
        for module in from_path.split('.'):
            from_model = getattr(from_model, module)
        if fix:
            for name, parameter in from_model.named_parameters():
                parameter.requires_grad_(False)
        for module in to_path.split('.')[:-1]:
            to_model = getattr(to_model, module)   
        setattr(to_model, to_path.split('.')[-1], from_model)
        

    def load_weight(self):

        def inner(weight_config):
            to_load_model = self._load_weight_file(
                weight_config["archive_file_path"],
                weight_config.get("weight_file_path", None)
            )
            from_paths = weight_config['from_module_path'].split(',')
            to_paths = weight_config['to_module_path'].split(',')
            fixs = weight_config['fix'].split(',')
            assert len(from_paths) == len(to_paths)
            for from_path, to_path, fix in zip(from_paths, to_paths, fixs):
                self._locate_and_replace(from_path, to_path, to_load_model, bool(fix))
        if self._load_weight_configs is None: return
        for config in self._load_weight_configs:
            inner(config)