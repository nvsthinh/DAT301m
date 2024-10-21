import argparse
import torch
import os
from tqdm.auto import tqdm
from models.seqGenSQL import SeqGenSQL
from utils import train_dataloader, val_dataloader, configure_optimizers, seed_everything

def contruct_params(parser):
    parser.add_argument('--data_dir', default="data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--model_name_or_path", default="t5-base")
    parser.add_argument("--max_seq_length", default=512)
    parser.add_argument("--max_output_length", default=200)
    parser.add_argument("--learning_rate", default=2e-4)
    parser.add_argument("--weight_decay", default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--warmup_steps", default=0)
    parser.add_argument("--train_batch_size", default=32)
    parser.add_argument("--eval_batch_size", default=32)
    parser.add_argument("--num_train_epochs", default=30)
    parser.add_argument("--gradient_accumulation_steps", default=16)
    parser.add_argument("--device", default=True)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--num_of_workers", default=4)
    args = parser.parse_args()

    return args

def train(model, train_loader, optimizer, scheduler):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        loss, _ = model.step(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            loss, _ = model.step(batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == '__main__':
    # Add parameters
    parser = argparse.ArgumentParser()
    args = contruct_params(parser)

    # Set seed
    seed_everything(args.seed)

    # Data
    train_loader = train_dataloader(args)
    eval_loader = val_dataloader(args)

    # Model initialization
    model = SeqGenSQL(args)

    # Optimizer and scheduler
    optimizer, scheduler = configure_optimizers(args, model)

    # Training Loop
    for epoch in range(args.num_train_epochs):
        print(f'Epoch {epoch+1}/{args.num_train_epochs}')

        # Training step
        train(model, train_loader, optimizer, scheduler)

        # Evaluation step
        eval_loss = evaluate(model, eval_loader)
        print(f"Evaluation Loss: {eval_loss}")
        # Save checkpoint
        model_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")