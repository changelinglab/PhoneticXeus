"""Config for the PhoneticXeus phone-recognition model (trust_remote_code)."""

from transformers import PretrainedConfig


class PhoneticXeusConfig(PretrainedConfig):
    """Configuration for PhoneticXeus.

    The architecture is defined by the bundled ESPnet-style YAML; this config
    only stores pointers to the vendored resources plus the few build options
    the recipe exposes at inference time.
    """

    model_type = "phoneticxeus"

    def __init__(
        self,
        config_yaml: str = "pxeus/model/xeusphoneme/resources/xeus_config.yaml",
        vocab_file: str = "pxeus/model/xeusphoneme/resources/ipa_vocab.json",
        interctc_layer_idx: list = (4, 8, 12),
        interctc_use_conditioning: bool = True,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        self.config_yaml = config_yaml
        self.vocab_file = vocab_file
        self.interctc_layer_idx = list(interctc_layer_idx)
        self.interctc_use_conditioning = interctc_use_conditioning
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)
