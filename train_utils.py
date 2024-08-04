import argparse
import json
import os
import csv
import traceback
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
import accelerate

from amt.config import load_model_config
from src.model import ModelConfig
def train_loop(
    dataloader: DataLoader,
    _epoch: int,
    device,
    model,
    loss_fn,
    optimizer,
    accelerator,
    _resume_step: int = 0,
    global_step: int = 0,
    steps_per_checkpoint: int = 1000,
    scheduler=None,
    project_dir: str = None,
    accumulation_steps: int = 2,
    max_grad_norm: float = 1.0,  # Clipping value
    logger=None,
    writer=None
):
    avg_train_loss = 0
    trailing_loss = 0
    loss_buffer = []
    TRAILING_LOSS_STEPS = 100

    try:
        lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
    except Exception:
        lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

    model.train()
    grad_norm = 0.0

    optimizer.zero_grad(set_to_none=True)
    for __step, batch in (
        pbar := tqdm(
            enumerate(dataloader),
            total=len(dataloader) + _resume_step,
            initial=_resume_step,
            leave=False,
            desc=f"Training Epoch {_epoch}"
        )
    ):
        step = __step + _resume_step + 1
        enc_input, src, tgt, idxs = batch
        enc_input = enc_input.to(device)
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.cuda.amp.autocast():
            logits = model(enc_input, src)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt) / accumulation_steps  # Scale loss by accumulation steps

        accelerator.backward(loss)

        if (step + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        loss_buffer.append(loss.item() * accumulation_steps)
        trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(loss_buffer[-TRAILING_LOSS_STEPS:])
        avg_train_loss = sum(loss_buffer) / len(loss_buffer)

        logger.debug(
            f"EPOCH {_epoch} STEP {step}: "
            f"lr={lr_for_print}, "
            f"loss={round(loss.item() * accumulation_steps, 4)}, "
            f"trailing_loss={round(trailing_loss, 4)}, "
            f"average_loss={round(avg_train_loss, 4)}, "
            f"grad_norm={round(grad_norm, 4)}"
        )
        writer.add_scalar('train_loss', loss.item() * accumulation_steps, global_step)
        writer.add_scalar('grad_norm', grad_norm, global_step)
        writer.flush()

        pbar.set_postfix_str(
            f"lr={lr_for_print}, "
            f"loss={round(loss.item() * accumulation_steps, 4)}, "
            f"trailing={round(trailing_loss, 4)}, "
            f"grad_norm={round(grad_norm, 4)}"
        )

        if scheduler:
            scheduler.step()
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

        if steps_per_checkpoint and step % steps_per_checkpoint == 0:
            make_checkpoint(accelerator, _epoch, step, project_dir, logger)

        global_step += 1

    logger.info(
        f"EPOCH {_epoch}: Finished training - "
        f"average_loss={round(avg_train_loss, 4)}"
    )

    return avg_train_loss, global_step

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_train_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--spc", type=int, default=1000, help="Steps per checkpoint")
    parser.add_argument("--pdir", type=str, required=True, help="Project directory")
    return parser.parse_args()

def parse_resume_args():
    parser = argparse.ArgumentParser(description="Resume training arguments")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--cdir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--rstep", type=int, required=True, help="Resume step")
    parser.add_argument("--repoch", type=int, required=True, help="Resume epoch")
    parser.add_argument("--spc", type=int, default=1000, help="Steps per checkpoint")
    parser.add_argument("--pdir", type=str, required=True, help="Project directory")
    parser.add_argument("--resume_mode", type=str, required=True, choices=["pt", "ft"], help="Resume mode")
    return parser.parse_args()

    
def _train(
    epochs: int,
    accelerator: Accelerator,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device,
    loss_fn,
    logger,
    writer,
    steps_per_checkpoint: int = 1000,
    resume_step: int | None = None,
    resume_epoch: int | None = None,
    project_dir: str = None,
):
    start_epoch = 0
    global_step = 0

    if resume_epoch is not None:
        start_epoch = resume_epoch + 1

    if project_dir:
        if not os.path.exists(os.path.join(project_dir, "checkpoints")):
            os.makedirs(os.path.join(project_dir, "checkpoints"))

    if accelerator.is_main_process:
        loss_csv = open(os.path.join(project_dir, "loss.csv"), "w")
        epoch_csv = open(os.path.join(project_dir, "epoch.csv"), "w")
        loss_writer = csv.writer(loss_csv)
        epoch_writer = csv.writer(epoch_csv)
        loss_writer.writerow(["epoch", "step", "loss"])
        epoch_writer.writerow(["epoch", "avg_train_loss", "avg_val_loss", "avg_val_loss_aug"])

    if resume_step is not None:
        assert resume_epoch is not None, "Must provide resume epoch"
        logger.info(f"Resuming training from step {resume_step} - logging as EPOCH {resume_epoch}")
        skipped_dataloader = accelerator.skip_first_batches(dataloader=train_dataloader, num_batches=resume_step)

        avg_train_loss, global_step = train_loop(
            dataloader=skipped_dataloader,
            _epoch=resume_epoch,
            device=device,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accelerator=accelerator,
            _resume_step=resume_step,
            global_step=global_step,
            steps_per_checkpoint=steps_per_checkpoint,
            scheduler=scheduler,
            project_dir=project_dir,
            logger=logger,
            writer=writer
        )
        avg_val_loss = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, device=device, model=model, loss_fn=loss_fn, logger=logger, aug=False
        )
        avg_val_loss_aug = val_loop(
            dataloader=val_dataloader, _epoch=resume_epoch, device=device, model=model, loss_fn=loss_fn, logger=logger, aug=True
        )
        if accelerator.is_main_process:
            epoch_writer.writerow([resume_epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug])
            epoch_csv.flush()
            make_checkpoint(accelerator, start_epoch, 0, project_dir, logger)

    for epoch in range(start_epoch, epochs + start_epoch):
        try:
            avg_train_loss, global_step = train_loop(
                dataloader=train_dataloader,
                _epoch=epoch,
                device=device,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accelerator=accelerator,
                global_step=global_step,
                steps_per_checkpoint=steps_per_checkpoint,
                scheduler=scheduler,
                project_dir=project_dir,
                logger=logger,
                writer=writer
            )
            avg_val_loss = val_loop(
                dataloader=val_dataloader, _epoch=epoch, device=device, model=model, loss_fn=loss_fn, logger=logger, aug=False
            )
            avg_val_loss_aug = val_loop(
                dataloader=val_dataloader, _epoch=epoch, device=device, model=model, loss_fn=loss_fn, logger=logger, aug=True
            )
            if accelerator.is_main_process:
                epoch_writer.writerow([epoch, avg_train_loss, avg_val_loss, avg_val_loss_aug])
                epoch_csv.flush()
                make_checkpoint(accelerator, epoch + 1, 0, project_dir, logger)
        except Exception as e:
            logger.debug(traceback.format_exc())
            raise e

    logging.shutdown()
    if accelerator.is_main_process:
        loss_csv.close()
        epoch_csv.close()


def train_loop(
    dataloader: DataLoader,
    _epoch: int,
    device,
    model,
    loss_fn,
    optimizer,
    accelerator,
    _resume_step: int = 0,
    global_step: int = 0,
    steps_per_checkpoint: int = 1000,
    scheduler=None,
    project_dir: str = None,
    accumulation_steps: int = 2,
    max_grad_norm: float = 1.0,  # Clipping value
    logger=None,
    writer=None
):
    avg_train_loss = 0
    trailing_loss = 0
    loss_buffer = []
    TRAILING_LOSS_STEPS = 100

    try:
        lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
    except Exception:
        lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

    model.train()
    grad_norm = 0.0

    optimizer.zero_grad(set_to_none=True)
    for __step, batch in (
        pbar := tqdm(
            enumerate(dataloader),
            total=len(dataloader) + _resume_step,
            initial=_resume_step,
            leave=False,
            desc=f"Training Epoch {_epoch}"
        )
    ):
        step = __step + _resume_step + 1
        enc_input, src, tgt, idxs = batch
        enc_input = enc_input.to(device)
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.cuda.amp.autocast():
            logits = model(enc_input, src)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt) / accumulation_steps  # Scale loss by accumulation steps

        accelerator.backward(loss)

        if (step + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        loss_buffer.append(loss.item() * accumulation_steps)
        trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(loss_buffer[-TRAILING_LOSS_STEPS:])
        avg_train_loss = sum(loss_buffer) / len(loss_buffer)

        logger.debug(
            f"EPOCH {_epoch} STEP {step}: "
            f"lr={lr_for_print}, "
            f"loss={round(loss.item() * accumulation_steps, 4)}, "
            f"trailing_loss={round(trailing_loss, 4)}, "
            f"average_loss={round(avg_train_loss, 4)}, "
            f"grad_norm={round(grad_norm, 4)}"
        )
        writer.add_scalar('train_loss', loss.item() * accumulation_steps, global_step)
        writer.add_scalar('grad_norm', grad_norm, global_step)
        writer.flush()

        pbar.set_postfix_str(
            f"lr={lr_for_print}, "
            f"loss={round(loss.item() * accumulation_steps, 4)}, "
            f"trailing={round(trailing_loss, 4)}, "
            f"grad_norm={round(grad_norm, 4)}"
        )

        if scheduler:
            scheduler.step()
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

        if steps_per_checkpoint and step % steps_per_checkpoint == 0:
            make_checkpoint(accelerator, _epoch, step, project_dir, logger)

        global_step += 1

    logger.info(
        f"EPOCH {_epoch}: Finished training - "
        f"average_loss={round(avg_train_loss, 4)}"
    )

    return avg_train_loss, global_step

@torch.no_grad()
def val_loop(
    dataloader: DataLoader,
    _epoch: int,
    device,
    model,
    loss_fn,
    logger,
    aug: bool
):
    loss_buffer = []
    model.eval()

    for step, batch in (
        pbar := tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            leave=False,
            desc=f"Evaluating Epoch {_epoch}"
        )
    ):
        enc_input, src, tgt, idxs = batch
        enc_input = enc_input.to(device)
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.cuda.amp.autocast():
            logits = model(enc_input, src)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

        loss_buffer.append(loss.item())
        avg_val_loss = sum(loss_buffer) / len(loss_buffer)
        pbar.set_postfix_str(f"average_loss={round(avg_val_loss, 4)}")

    logger.info(
        f"EPOCH {_epoch}: Finished evaluation "
        f"{'(aug)' if aug else ''} - "
        f"average_loss={round(avg_val_loss, 4)}"
    )

    return avg_val_loss


def setup_scheduler(optimizer, num_epochs, steps_per_epoch, warmup_steps=1000, end_ratio=0.1):
    total_steps = num_epochs * steps_per_epoch
    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1,
        total_iters=warmup_steps,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=total_steps - warmup_steps,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup_steps],
    )
    return lr_scheduler

def make_checkpoint(_accelerator, _epoch: int, _step: int, project_dir: str, logger):
    checkpoint_dir = os.path.join(
        project_dir,
        "checkpoints",
        f"epoch{_epoch}_step{_step}",
    )
    logger.info(f"EPOCH {_epoch}: Saving checkpoint - {checkpoint_dir}")
    _accelerator.save_state(checkpoint_dir)

def get_max_norm(named_parameters):
    max_grad_norm = {"val": 0.0}
    for name, parameter in named_parameters:
        if parameter.grad is not None and parameter.requires_grad:
            grad_norm = parameter.grad.data.norm(2).item()
            if grad_norm >= max_grad_norm["val"]:
                max_grad_norm["name"] = name
                max_grad_norm["val"] = grad_norm
    return max_grad_norm