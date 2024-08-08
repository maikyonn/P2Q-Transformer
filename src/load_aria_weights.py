
from amt.config import load_model_config
from amt.model import ModelConfig
from aria.tokenizer import AbsTokenizer

from amt.model import ModelConfig
from aria.utils import _load_weight
from amt.config import load_model_config

from aria.tokenizer import AbsTokenizer

from src.model import P2QTransformer 

def remap_aria_keys(model_state, ignore):
    new_model_state = {}
    for key, value in model_state.items():
        new_key = key.replace('_orig_mod.', '')
        if 'encoder' in new_key or new_key in ignore:
            # print("Did not load in " + new_key)
            continue
        # print('Loaded in ' + new_key)
        new_model_state[new_key] = value
    return new_model_state

def get_p2q(config, tokenizer, weights_path=None, requires_mapping=False):
    M_CONFIG = "medium-stacked"
    model = P2QTransformer(tokenizer.vocab_size, config['seq_len'], config['d_model'], config['n_heads'], config['n_layers'])

    if weights_path:
        loaded_weights = _load_weight(weights_path, device="cuda")
        loaded_weights = {k: v for k, v in loaded_weights.items() if "rotary_emb" not in k}

        if requires_mapping:
            ignore = ["decoder.token_embedding.weight", "decoder.output.weight"]
            loaded_weights = remap_aria_keys(loaded_weights, ignore)
        model_dict = model.state_dict()
        model_dict.update(loaded_weights)
        model.load_state_dict(model_dict)
    return model
