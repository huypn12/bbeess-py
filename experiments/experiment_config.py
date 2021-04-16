from abc import ABC, abstractmethod
import numpy as np


class ExperimentConfig(object):
    @staticmethod
    @abstractmethod
    def _get_config():
        """Overriden by subclass"""
        raise NotImplementedError()

    @classmethod
    def get_config(cls, model_name: str):
        cfg = cls._get_config()
        if model_name not in cfg:
            raise ValueError(f"Configuration not found for entry {model_name}")
        return cfg[model_name]

    @classmethod
    def get_all_config_names(cls):
        cfg = cls._get_config()
        return list(cfg.keys())