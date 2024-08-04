
from amt.config import load_model_config
from amt.model import ModelConfig
from aria.tokenizer import AbsTokenizer

from amt.model import ModelConfig
from aria.utils import _load_weight
from amt.config import load_model_config

from aria.tokenizer import AbsTokenizer

from src.model import P2QTransformer 


def get_p2q(tokenizer, with_weights=False ):
    if with_weights:
        M_PATH = "piano-medium-stacked-1.0.safetensors"
        M_CONFIG = "medium-stacked"
        tokenizer = AbsTokenizer()

        loaded_aria_weights = _load_weight(M_PATH, device="cuda")
        loaded_aria_weights = {k: v for k, v in loaded_aria_weights.items() if "rotary_emb" not in k}
        model_config = ModelConfig(**load_model_config(M_CONFIG))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = P2QEncoderDecoder(model_config)

        ignore = ["decoder.token_embedding.weight", "decoder.output.weight"]
        def remap_keys(model_state):
            new_model_state = {}
            for key, value in model_state.items():
                new_key = key.replace('_orig_mod.', '')
                if 'encoder' in new_key or new_key in ignore:
                    # print("Did not load in " + new_key)
                    continue
                # print('Loaded in ' + new_key)
                new_model_state[new_key] = value
            return new_model_state

        mapped_loaded_model_state = remap_keys(loaded_aria_weights)
        model_dict = model.state_dict()
        model_dict.update(mapped_loaded_model_state)
        model.load_state_dict(model_dict)
        return model
    else:
        M_CONFIG = "medium-stacked"
        model_config = ModelConfig(**load_model_config(M_CONFIG))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = P2QTransformer(model_config)
        return model