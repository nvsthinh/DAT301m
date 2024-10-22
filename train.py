import argparse
import torch
import os
from tqdm.auto import tqdm
from models.seqGenSQL import SeqGenSQL
import warnings
from utils import train_dataloader, val_dataloader, configure_optimizers, seed_everything, step

warnings.filterwarnings('ignore')

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
    parser.add_argument("--train_batch_size", default=16)
    parser.add_argument("--eval_batch_size", default=8)
    parser.add_argument("--num_train_epochs", default=30)
    parser.add_argument("--gradient_accumulation_steps", default=16)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--num_of_workers", default=4)

    # Data Augmentation and model enhancement Options
    parser.add_argument("--include_data_type", default=True)
    parser.add_argument("--num_sample_rows", default=3)
    parser.add_argument("--data_aug", default=[], help="List, use one of these options: ['select_column', 'where_value']. Default is []")
    parser.add_argument("--use_modified_network", default=True, help="Use gated layer to decide whether to extract or to generate")
    parser.add_argument("--generated_data_files", default=[], help="List of the generated data files. Default is []")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args

def train(model, train_loader, optimizer, scheduler, args):
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        loss, _ = step(model, batch, args)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, val_loader, args):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            loss, _ = step(model, batch, args)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == '__main__':
    # Add parameters
    parser = argparse.ArgumentParser()
    args = contruct_params(parser)

    # Set seed
    seed_everything(args.seed)

    # Data
    print('DataLoader: ....')
    train_loader = train_dataloader(args)
    eval_loader = val_dataloader(args)
    print('Complete !!!')

    # Model initialization
    print('Model: ....')
    model = SeqGenSQL(args)
    print('Complete !!!')

    # Optimizer and scheduler
    optimizer, scheduler = configure_optimizers(args, model, train_loader)

    # Training Loop
    for epoch in range(args.num_train_epochs):
        print(f'Epoch {epoch+1}/{args.num_train_epochs}')

        # Training step
        train(model, train_loader, optimizer, scheduler, args)

        # Evaluation step
        eval_loss = evaluate(model, eval_loader, args)
        print(f"Evaluation Loss: {eval_loss}")
        # Save checkpoint
        model_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")