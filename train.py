import json
import os
import csv
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback
from aria.tokenizer import AbsTokenizer
from src.dataset import get_ds
from src.load_aria_weights import get_p2q
from accelerate import Accelerator
from safetensors.torch import save_file
from train_utils import _train, load_config, setup_scheduler


def main(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, vocab_size, tokenizer = get_ds(
        config['train_txt_path'], config['val_txt_path'], config['block_size'], train_size_limit=config['train_size_limit'], batch_size=config['batch_size'], num_workers=config['workers'])
    model = get_p2q(tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['initial_lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.encode([('<P>')]), label_smoothing=0.1).to(device)
    writer = SummaryWriter()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    accelerator = Accelerator(mixed_precision='fp16')
    num_epochs = config['epochs']
    steps_per_epoch = len(train_dataloader)
    mode = config['mode']

    if mode not in ["pretrain", "finetune", "resume"]:
        print("Unrecognized mode")
        return

    scheduler = setup_scheduler(optimizer, num_epochs, steps_per_epoch)

    if mode == "resume":
        _train(
            epochs=num_epochs,
            accelerator=accelerator,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            logger=logger,
            writer=writer,
            steps_per_checkpoint=config['steps_per_checkpoint'],
            resume_step=config['resume_step'],
            resume_epoch=config['resume_epoch'],
            project_dir=config['output_dir']
        )
    else:
        _train(
            epochs=num_epochs,
            accelerator=accelerator,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            logger=logger,
            writer=writer,
            steps_per_checkpoint=config['steps_per_checkpoint'],
            resume_step=None,
            resume_epoch=None,
            project_dir=config['output_dir']
        )

if __name__ == "__main__":
    config_path = 'train_config.json'
    main(config_path)
