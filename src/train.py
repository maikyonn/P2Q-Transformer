import torch
import torch.optim
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from dataset import get_ds
from model import get_model

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, vocab_size, tokenizer = get_ds(config)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.encode(['<SEP>'])[0], label_smoothing=0.1).to(device)
    model = get_model(config, vocab_size, vocab_size).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    writer = SummaryWriter()


    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()

            global_step += 1

if __name__ == "__main__":
    main()