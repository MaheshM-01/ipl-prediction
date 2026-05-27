# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Iterable,Optional,Union

# from omegaconf import DictConfig, OmegaConf

# class ConfigError(ValueError):
#     "raised when the configuration file is invalid or not found "
#     pass

# class ConfigLoader:
#     def __init__(self,config_path: Optional[Union[str,Path]]=None)-> None:
#         root=ConfigLoader.get_root()
#         self.config_path=Path(config_path) if config_path else root/'config'/'config.yaml'
#         self.config_path=self.config_path.resolve()

#         @staticmethod
#         def get_root()->Path:
#             return Path(__file__).resolve().parent.parent.parent
        
#         def load(self)-> DictConfig:
#             if not self.config_path.exists():
#                 raise FileNotFoundError(f' config file is not fount at the : {self.config_path}')
            
#             cfg=OmegaConf.load(self.config_path)
#             self._validate(cfg)
#             cfg=self._resolve_paths(cfg)
#             return cfg
        
#         def _validate(self,cfg: DictConfig)-> None:
#             self._ensure_keys(cfg,('paths','project'),'root')
#             self._ensure_keys(cfg.paths,('raw_matches','raw_deliveries','processed_data','models_dir'),'paths')

#         def _resolve_paths(self,cfg: DictConfig)-> DictConfig:
#             root=self.get_root()

#             for key in ('raw_matches','raw_deliveries','processed_data','models_dir','logs_dir'):
#                 if key in cfg.paths:
#                     raw_path=Path(str(cfg.paths[key]))
#                     abs_path=raw_path if raw_path.is_absolute() else (root/raw_path).resolve()
#                     cfg.paths[key]=str(abs_path)

#             for key in ('processed_data','models_dir','logs_dir'):

#                 Path(str(cfg.paths[key])).mkdir(parents=True,exist_ok=True)

#             for key in ('raw_matches','raw_deliveries'):
#                 p=Path(str(cfg.paths[key]))
#                 if not p.exists():
#                     raise FileNotFoundError(f' raw ipl source file  missing at : {p}')
                

            
            
#             return cfg
        
#         @staticmethod
#         def _ensure_keys(container: Any,keys: Iterable[str],section_name: str)->None:
#             missing=[key for key in keys if key not in container]

#             if  missing:
#                 raise ConfigError(f' missing keys in {section_name} : {missing}')
            


from __future__ import annotations
import os
from typing import Dict,Any

from pathlib import Path
from typing import Iterable, Optional, Union
from omegaconf import DictConfig, OmegaConf

class ConfigError(ValueError):
    """Raised when the configuration file is invalid or not found."""
    pass

class ConfigLoader:
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        # Static method call directly using Class Reference
        root = ConfigLoader.get_root() 
        self.config_path = Path(config_path) if config_path else root / 'config' / 'config.yaml'
        self.config_path = self.config_path.resolve()

    @staticmethod
    def get_root() -> Path:
        # Resolves path accurately from: src/config/config.py -> parent(config) -> parent(src) -> parent(root)
        return Path(__file__).resolve().parent.parent

    def load(self) -> DictConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f'Config file is not found at: {self.config_path}')
        
        cfg = OmegaConf.load(self.config_path)
        self._validate(cfg)
        cfg = self._resolve_paths(cfg)
        return cfg
    
    def _validate(self, cfg: DictConfig) -> None:
        self._ensure_keys(cfg, ('paths', 'project'), 'root')
        self._ensure_keys(cfg.paths, ('raw_matches', 'raw_deliveries', 'processed_data', 'models_dir'), 'paths')

    def _resolve_paths(self, cfg: DictConfig) -> DictConfig:
        root = ConfigLoader.get_root()

        # Dynamically map absolute path lookups
        for key in ('raw_matches', 'raw_deliveries', 'processed_data', 'models_dir', 'logs_dir'):
            if key in cfg.paths:
                raw_path = Path(str(cfg.paths[key]))
                abs_path = raw_path if raw_path.is_absolute() else (root / raw_path).resolve()
                cfg.paths[key] = str(abs_path)

        # Build execution directories frames automatically
        for key in ('processed_data', 'models_dir', 'logs_dir'):
            Path(str(cfg.paths[key])).mkdir(parents=True, exist_ok=True)

        # Verify physical file existence checks
        for key in ('raw_matches', 'raw_deliveries'):
            p = Path(str(cfg.paths[key]))
            if not p.exists():
                raise FileNotFoundError(f'Raw IPL source file missing at: {p}')
        
        return cfg
    
    @staticmethod
    def _ensure_keys(container: Any, keys: Iterable[str], section_name: str) -> None:
        missing = [key for key in keys if key not in container]
        if missing:
            raise ConfigError(f'Missing keys in configuration section [{section_name}]: {missing}')