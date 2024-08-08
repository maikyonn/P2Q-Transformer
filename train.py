
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
from src.dataset import get_ds
from src.load_aria_weights import get_p2q
from accelerate import Accelerator
from train_utils import _get_optim, _train, load_config
import wandb

def main(config_path):
    # Load config
    config = load_config(config_path)

    gradient_accumulation_steps=4

    # Initialize the Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16'  # Adjust based on your needs
    )

    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project=config['model'], config=config)

    # Initialize device
    device = accelerator.device

    # Get dataset and dataloaders
    train_dataloader, val_dataloader, vocab_size, tokenizer = get_ds(
        config['train_txt_path'], config['val_txt_path'], config['block_size'], train_size_limit=config['train_size_limit'], batch_size=config['batch_size'], num_workers=config['workers'])

    # Initialize model, optimizer, and loss function
    model = get_p2q(tokenizer).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.encode([('<P>')]), label_smoothing=0.1).to(device)
    writer = SummaryWriter()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    num_epochs = config['epochs']
    steps_per_epoch = len(train_dataloader) / gradient_accumulation_steps
    mode = config['mode']

    # Prepare the model, optimizer, and dataloaders with Accelerator
    optimizer, scheduler = _get_optim(config['initial_lr'], model, num_epochs, steps_per_epoch, config['warmup_steps'], config['end_ratio'])
    model, optimizer, train_dataloader, val_dataloader, scheduler= accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )


    if mode not in ["pretrain", "finetune", "resume"]:
        print("Unrecognized mode")
        return

    

    # Log the model and optimizer parameters only on the main process
    if accelerator.is_main_process:
        wandb.watch(model, log='all')

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
    
    # Finish the wandb run only on the main process
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    config_path = 'train_config.json'
    main(config_path)
