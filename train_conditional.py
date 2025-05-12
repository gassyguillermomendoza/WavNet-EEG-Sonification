import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import generate_conditional 
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device, num_epochs=100, resume_ckpt=None):
    os.makedirs("checkpoints", exist_ok=True)
    losses = []

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        model.load_state_dict(torch.load(resume_ckpt))
        print(f"Resumed from checkpoint: {resume_ckpt}")
        start_epoch = int(''.join(filter(str.isdigit, resume_ckpt)))

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for x, y, inst, pitch in train_loader:
            x = x.to(device).long()
            y = y.to(device).long()
            inst = inst.to(device).long()
            pitch = pitch.to(device).long()

            x_input = F.one_hot(x, num_classes=256).float().permute(0, 2, 1)  # (B, 256, T)

            logits = model(x_input, inst, pitch)  # (B, 256, T)
            logits = logits.permute(0, 2, 1).reshape(-1, 256)
            y = y.reshape(-1)

            print("target y:", y.min().item(), y.max().item())


            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
def train_with_logging(model, train_loader, optimizer, criterion, device, num_epochs,
                       checkpoint_dir="/cs/cs152/shared/gmendoza/checkpoints",
                       output_dir="outputs", resume_ckpt=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = []

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        last_outputs = None  # For token histogram
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for x, y, inst, pitch in progress_bar:
            x, y = x.to(device), y.to(device)
            inst, pitch = inst.to(device), pitch.to(device)

            with torch.no_grad():
                y_min, y_max = y.min().item(), y.max().item()
                if y_min < 0 or y_max > 255:
                    print(f"Invalid target: y.min={y_min}, y.max={y_max}")

            x = F.one_hot(x, num_classes=256).float().permute(0, 2, 1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(x, inst, pitch)
                loss = criterion(outputs.permute(0, 2, 1).reshape(-1, 256), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            last_outputs = outputs

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch} Avg Loss: {epoch_loss:.4f}")

        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

        with torch.no_grad():
            log_probs = F.log_softmax(last_outputs, dim=1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=1).mean().item()
            print(f"Epoch {epoch} - Average Entropy: {entropy:.2f}")

            predicted_tokens = probs.argmax(dim=1).flatten().cpu().numpy()

            plt.figure(figsize=(6, 3))
            plt.hist(predicted_tokens, bins=range(257), density=True,
                     color='skyblue', edgecolor='black')
            plt.title(f"Predicted Token Distribution (Epoch {epoch})")
            plt.xlabel("Token ID")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"token_dist_epoch{epoch}.png"))
            plt.close()

        ckpt_path = os.path.join(checkpoint_dir, f"cond_wavenet_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")