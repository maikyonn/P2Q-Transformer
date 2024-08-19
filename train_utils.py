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
import wandb
from amt.config import load_model_config
from src.model import ModelConfig


from aria.data.midi import MidiDict
from aria.tokenizer import AbsTokenizer

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

        with torch.amp.autocast('cuda'):
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

        if accelerator.is_main_process:
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

    if accelerator.is_main_process:
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

def single_greedy_search(model, tokenizer, enc_input, max_length, device):

    start_token = tokenizer.encode(['<S>'])[0]
    end_token = tokenizer.encode(['<E>'])[0]
    pad_token = tokenizer.encode(['<P>'])[0]
    
    sequences = [start_token]  # Initialize with the start token

    for pos in range(max_length):
        chunk = enc_input[pos:pos + 1096]
        
        if len(chunk) < max_length:
            # Padding the last chunk if it's smaller than the window size
            chunk = chunk + [pad_token] * (max_length - len(chunk))
        
        chunk_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
        
        encoder_output = model.encode(chunk_tensor)

        current_length = len(sequences)
        padded_sequences = sequences + [pad_token] * (max_length - current_length)
        
        # Convert the sequence list into a tensor wiaath the correct shape
        input_tensor = torch.tensor(padded_sequences, device=device).unsqueeze(0)  # Adding batch dimension

        with torch.amp.autocast('cuda'):
            logits = model.logits(input_tensor, encoder_output)
        
        logits = logits[:, -1, :]  # Take the logits of the last token
        next_token = torch.argmax(logits, dim=-1).item()  # Choose the token with the highest probability
        
        sequences.append(next_token)  # Append the chosen token to the sequence
        
        if next_token == end_token:  # Stop if the end token is generated
            break
    
    return sequences

def test_inference(model, quant_path, perf_path, output_path, device, epoch):
    tokenizer = AbsTokenizer()
    midi_dict = MidiDict.from_midi(quant_path)
    tokenized_midi = tokenizer._tokenize_midi_dict(midi_dict=midi_dict)
    encoded_midi_seq = tokenizer.encode(tokenized_midi)
    

    val_midi_dict = MidiDict.from_midi(perf_path)
    val_tokenized_midi = tokenizer._tokenize_midi_dict(midi_dict=val_midi_dict)

    decoded_seq = single_greedy_search(model, tokenizer, encoded_midi_seq, 256, device)
    raw_output = tokenizer.decode(decoded_seq)
    # Append the outputs and epoch number to a file
    with open(output_path, 'a') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write("Tokenized MIDI Sequence:\n")
        f.write(str(tokenized_midi[1:256]) + '\n\n')
        f.write("Raw Output:\n")
        f.write(str(raw_output[:256]) + '\n')
        f.write("Actual Output:\n")
        f.write(str(val_tokenized_midi[:256]) + '\n')
        f.write("="*50 + "\n")  # Separator for clarity

    return tokenized_midi, raw_output



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
        epoch_writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    if resume_step is not None:
        assert resume_epoch is not None, "Must provide resume epoch"
        logger.info(f"Resuming training from step {resume_step} - logging as EPOCH {resume_epoch}")
        skipped_dataloader = accelerator.skip_first_batches(dataloader=train_dataloader, num_batches=resume_step)

        test_inference(model, midi_path="datasets/paired-dataset-5/performance/audio-https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DL2KdUIeYTQo.mid", output_path='./inf.txt', device=device)

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
            dataloader=val_dataloader, _epoch=resume_epoch, accelerator=accelerator, device=device, model=model, loss_fn=loss_fn, logger=logger, aug=False
        )
        if accelerator.is_main_process:
            epoch_writer.writerow([resume_epoch, avg_train_loss, avg_val_loss])
            epoch_csv.flush()
            make_checkpoint(accelerator, start_epoch, 0, project_dir, logger)

    for epoch in range(start_epoch, epochs + start_epoch):
        try:
            test_inference(model, quant_path='datasets/paired-dataset-5/quantized/Papillons op2.mid', perf_path='datasets/paired-dataset-5/performance/audio-https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D5csG23b4k9U.mid', output_path='./inf-val.txt', device=device, epoch=epoch)
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
                dataloader=val_dataloader, _epoch=epoch, device=device, accelerator=accelerator, model=model, loss_fn=loss_fn, logger=logger, aug=False
            )
            if accelerator.is_main_process:
                epoch_writer.writerow([epoch, avg_train_loss, avg_val_loss])
                epoch_csv.flush()
                make_checkpoint(accelerator, epoch + 1, 0, project_dir, logger)
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "avg_train_loss": avg_train_loss,
                    "avg_val_loss": avg_val_loss,
                })

        except Exception as e:
            logger.debug(traceback.format_exc())
            raise e

    # logging.shutdown()
    if accelerator.is_main_process:
        loss_csv.close()
        epoch_csv.close()

@torch.no_grad()
def val_loop(
    dataloader: DataLoader,
    _epoch: int,
    accelerator,
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

        with torch.amp.autocast('cuda'):
            logits = model(enc_input, src)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

        loss_buffer.append(loss.item())
        avg_val_loss = sum(loss_buffer) / len(loss_buffer)
        pbar.set_postfix_str(f"average_loss={round(avg_val_loss, 4)}")

    if accelerator.is_main_process:
        logger.info(
            f"EPOCH {_epoch}: Finished evaluation "
            f"{'(aug)' if aug else ''} - "
            f"average_loss={round(avg_val_loss, 4)}"
        )

    return avg_val_loss

def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: int = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler

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