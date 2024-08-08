

from aria.tokenizer import AbsTokenizer
import torch


import os
import torch
import torch.optim
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer
from rich import progress

from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict

from typing import Callable, Iterable

from multiprocessing import Pool, get_start_method
import os
from functools import partial
from multiprocessing import Pool
from progress.bar import Bar  # Assuming you're using the progress package for the progress bar


import functools
import logging


def get_seqs(
    tokenizer: Tokenizer,
    midi_dict_iter: Iterable,
):
    num_proc = os.cpu_count()

    # Can't pickle geneator object when start method is spawn
    if get_start_method() == "spawn":
        logging.info(
            "Converting generator to list due to multiprocessing start method"
        )
        midi_dict_iter = [_ for _ in midi_dict_iter]

    with Pool(16) as pool:
        results = pool.imap(
            functools.partial(_get_seqs, _tokenizer=tokenizer), midi_dict_iter
        )

        yield from results

def _get_seqs(_entry: MidiDict | dict, _tokenizer: Tokenizer):
    logger = setup_logger()

    if isinstance(_entry, str):
        _midi_dict = MidiDict.from_msg_dict(json.loads(_entry.rstrip()))
    elif isinstance(_entry, dict):
        _midi_dict = MidiDict.from_msg_dict(_entry)
    elif isinstance(_entry, MidiDict):
        _midi_dict = _entry
    else:
        raise Exception

    try:
        _tokenized_seq = _tokenizer.tokenize(_midi_dict)
    except Exception as e:
        print(e)
        logger.info(f"Skipping midi_dict: {e}")
        return
    else:
        if _tokenizer.unk_tok in _tokenized_seq:
            logger.warning("Unknown token seen while tokenizing midi_dict")
        return _tokenized_seq
    
def setup_logger():
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def _format(tok):
    # This is required because json formats tuples into lists
    if isinstance(tok, list):
        return tuple(tok)
    return tok

def trunc_seq(seq: list, seq_len: int, pad_token):
    """Truncate or pad sequence to feature sequence length."""
    seq += [pad_token] * (seq_len - len(seq))

    return seq[:seq_len]

def get_combined_slices(data, slice_length, overlap=512):
    slices = []
    step = slice_length - overlap
    for start_idx in range(0, len(data), step):
        end_idx = start_idx + slice_length
        slice = data[start_idx:end_idx]
        if len(slice) == slice_length:
            slices.append(slice)
    return slices

def process_line(line, base_path, tokenizer, seq_len):
    quant_midi_path, perf_midi_path = line.strip().split('|')

    quant_midi_dict = MidiDict.from_midi(os.path.join(base_path, quant_midi_path))
    perf_midi_dict = MidiDict.from_midi(os.path.join(base_path, perf_midi_path))
    tokenized_quant_midi = tokenizer._tokenize_midi_dict(midi_dict=quant_midi_dict)
    tokenized_perf_midi = tokenizer._tokenize_midi_dict(midi_dict=perf_midi_dict)

    tuple_quant_tokenized_midi = [_format(tok) for tok in tokenized_quant_midi]
    tuple_perf_tokenized_midi = [_format(tok) for tok in tokenized_perf_midi]

    quant_combined_slices = get_combined_slices(tuple_quant_tokenized_midi, seq_len, overlap=128)
    perf_combined_slices = get_combined_slices(tuple_perf_tokenized_midi, seq_len, overlap=128)
    
    return quant_combined_slices, perf_combined_slices

class QPDataset(Dataset):
    def __init__(self, path: str, seq_len, size_limit, num_workers):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = AbsTokenizer(return_tensors=True)
        self.sos_token = self.tokenizer.encode(['<S>'])
        self.eos_token = self.tokenizer.encode(['<E>'])
        self.pad_token = self.tokenizer.encode(['<P>'])
        self.input_seq, self.target_seq = self.load_midi_data(path, size_limit)
        self.path = path
        self.id2tok = self.tokenizer.id_to_tok
        self.tok2id = self.tokenizer.tok_to_id


    def load_midi_data(self, path, limit=10000, num_workers=30):
        x = []
        y = []
        with open(path) as datafile:
            lines = datafile.readlines()
            lines = lines[:limit]
            
            base_path = "./datasets/paired-dataset-5/"
            
            with Pool(num_workers) as pool:
                process_func = partial(process_line, base_path=base_path, tokenizer=self.tokenizer, seq_len=self.seq_len)
                
                bar = Bar('Processing', max=len(lines))
                for quant_combined_slices, perf_combined_slices in pool.imap(process_func, lines):
                    x.extend(quant_combined_slices)
                    y.extend(perf_combined_slices)
                    bar.next()
                bar.finish()
        
        return x, y

    
    def __len__(self):
        return len(self.input_seq)
    
    def get_max_len_midi_data(self):
        x = []
        for seq in progress.track(self.input_seq):
            x.append(len(seq))
        return max(x)
    
    def __getitem__(self, idx: int):
        #[('prefix', 'instrument', 'piano'), '<S>', ('piano', 60, 105), ('onset', 0), ('dur', 4450)
        # -> [('piano', 60, 105), ('onset', 0), ('dur', 4450), ('piano', 51, 105)
        # Cut string to remove prefix and ending character.
        # input_midi = self.input_seq[idx][2:-1]
        # targ_midi = self.target_seq[idx][2:-1]
        input_midi = self.input_seq[idx]
        targ_midi = self.target_seq[idx]

        src = trunc_seq(
            seq=targ_midi,
            seq_len=self.seq_len,
            pad_token=self.pad_token
        )
        tgt = trunc_seq(
            seq=targ_midi[1:],
            seq_len=self.seq_len,
            pad_token=self.pad_token
        )

        return self.tokenizer.encode(input_midi), self.tokenizer.encode(src), self.tokenizer.encode(tgt), idx


def get_ds(train_txt_path, val_txt_path, block_size, train_size_limit, batch_size, num_workers):
    train_ds = QPDataset(train_txt_path, block_size, size_limit=train_size_limit, num_workers=num_workers)
    val_ds = QPDataset(val_txt_path, block_size, size_limit=train_size_limit, num_workers=num_workers)
    
    print("Longest Song Length: ", train_ds.get_max_len_midi_data())
    print("Number of data points in training set: ", len(train_ds))
    print("Number of data points in validation set: ", len(val_ds))
    print("Training with batch size: ", batch_size)
    print("Steps per epoch: ", len(train_ds) // batch_size)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, train_ds.tokenizer.vocab_size, train_ds.tokenizer


