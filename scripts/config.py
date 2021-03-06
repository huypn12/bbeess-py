from enum import Enum


class ConfigKey(Enum):
    is_bee_model = "is_bee_model"
    per_bscc_sampling = "per_bscc_sampling"
    abc_threshold_decreasing_factor = "abc_threshold_decreasing_factor"
    has_synthesis = "has_synthesis"


_config_dict = {
    ConfigKey.is_bee_model.value: False,
    ConfigKey.per_bscc_sampling.value: 1000,
    ConfigKey.abc_threshold_decreasing_factor.value: 1,
    ConfigKey.has_synthesis.value: True,
}


def has_synthesis():
    return _config_dict[ConfigKey.has_synthesis.value]


def set_has_synthesis(b: bool):
    _config_dict[ConfigKey.has_synthesis.value] = b


def is_bee_model():
    return _config_dict[ConfigKey.is_bee_model.value]


def per_bscc_sampling():
    return _config_dict[ConfigKey.per_bscc_sampling.value]


def abc_threshold_decreasing_factor():
    return _config_dict[ConfigKey.abc_threshold_decreasing_factor.value]


def set_bee_model(b: bool):
    _config_dict[ConfigKey.is_bee_model.value] = b


def unset_bee_model():
    _config_dict[ConfigKey.is_bee_model.value] = True


def set_per_bscc_sampling(n: int):
    _config_dict[ConfigKey.per_bscc_sampling.value] = n


def unset_per_bscc_sampling():
    _config_dict[ConfigKey.per_bscc_sampling.value] = 1000


def set_abc_threshold_decreasing_factor(factor: float):
    _config_dict[ConfigKey.abc_threshold_decreasing_factor.value] = factor


def unset_abc_threshold_decreasing_factor():
    _config_dict[ConfigKey.abc_threshold_decreasing_factor.value] = 1