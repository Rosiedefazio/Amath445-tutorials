import os
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)

model_default_args = dict(
    in_channels=1,           # 1 for MNIST
    patch_size=7,
    emb_size=64,
    n_heads=8,
    n_layers=6,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb_project = "MyProject"


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_args_parser():
    parser = argparse.ArgumentParser("Training and evaluation script", add_help=False)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--n_cpu", default=8, type=int)
    parser.add_argument("--input_dir", default="./data/", type=str)
    parser.add_argument("--compile", default=False, type=boolean_string)
    parser.add_argument(
        "--output_dir",
        default="model_output",
        help="all model outputs will be stored in this dir",
    )
    parser.add_argument("--resume_training", default=False, type=boolean_string)
    parser.add_argument(
        "--saved_model_path", default="./model_output/saved_models", type=str
    )

    return parser


def conf_matrix_plot(cf_matrix: np.ndarray, title: str = ""):
    """
    Return matplotlib fig of confusion matrix
    """
    fig, axs = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=axs,
    )
    fig.suptitle(title)
    return fig


def evaluation_metrics(y_true: np.ndarray, y_preds: np.ndarray):
    conf_matrix = confusion_matrix(y_true, y_preds)
    accuracy, f1, precision, recall = (
        accuracy_score(y_true, y_preds),
        f1_score(y_true, y_preds, zero_division=0.0, average="macro"),
        precision_score(y_true, y_preds, zero_division=0.0, average="macro"),
        recall_score(y_true, y_preds, zero_division=0.0, average="macro"),
    )
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }, conf_matrix


class MNISTDataset(Dataset):
    def __init__(self, path):
        self.path = path  # path to the dataset
        self.data = os.listdir(path)  # list of all the files in the dataset
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),  # convert the image to grayscale
                transforms.Resize((28, 28)),  # resize the image to 28x28
                transforms.ToTensor(),  # convert the image to a PyTorch tensor
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.data[idx]))
        img = self.transform(img)
        label = int(self.data[idx].split("__")[1].split(".")[0])
        return {"img": img, "label": label}


@torch.inference_mode()
def perform_inference(model, dataloader, device: str, loss_fn=None):
    """
    Perform inference on given dataset using given model on the specified device. If loss_fn is
     provided, it also computes the loss and returns [y_preds, y_true, losses].
    """
    model.eval()  # Set the model to evaluation mode, this disables training specific operations
    y_preds = []
    y_true = []
    losses = []

    print("[inference.py]: Running inference...")
    for i, batch in tqdm(enumerate(dataloader)):
        inputs = batch["img"].to(device)
        outputs = model(inputs)
        if loss_fn is not None:
            labels = batch["label"].to(device)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())

        preds = F.softmax(outputs.detach().cpu(), dim=1).argmax(dim=1)
        y_preds.append(preds.numpy())

    model.train()  # Set the model back to training mode
    y_true, y_preds = np.concatenate(y_true), np.concatenate(y_preds)
    return y_true, y_preds, np.mean(losses) if losses else None


def init_weights(module):
    """
    Initialise weights of given module using Kaiming Normal initialisation for linear and
    convolutional layers, and zeros for bias.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Head(nn.Module):
    def __init__(self, head_size, n_embed, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) #nn.linear is a linear layer of a NN which is a 2d matrix 
        self.value = nn.Linear(n_embed, head_size, bias=False) #bias is false because we dont want it 
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout) #dropout is here just cuz, standard practice tbh 

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        out = F.softmax(q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5, dim=-1) #softmax is a generalization of sigmoid 
        out = self.dropout(out) #transpose transponses the needed values in the matrix, the other is batch size 
        out = out @ v 
        return out

# y(x) = sum(H*W)
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_heads, n_embed, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #dim = -1 to avoid batched 
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module): #2 layer neural network
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads, n_embed, dropout=0.2):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(head_size, n_heads, n_embed, dropout)
        self.ffwd = MLP(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x)) #residual neural netweork because there is a skip connection which is adding x to itslef 
        x = x + self.ffwd(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        n_patches,
        emb_size,
        n_heads,
        n_layers,
        n_classes,
        class_freqs,
        dropout=0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_heads, emb_size, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

        self.head.bias = nn.Parameter(torch.log(class_freqs))
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
#transformers help retain some informaiton about how parts of the CNN connect to each other 

def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, time)
    os.makedirs(os.path.join(args.output_dir, "saved_models"), exist_ok=True)

    train_dataset = MNISTDataset(os.path.join(args.input_dir, "train"))
    valid_dataset = MNISTDataset(os.path.join(args.input_dir, "valid"))
    test_dataset = MNISTDataset(os.path.join(args.input_dir, "test"))
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.n_cpu, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.n_cpu, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.n_cpu, shuffle=False
    )

    epoch_start, epoch_end = 0, args.n_epochs
    best_loss_val = float("inf")
    class_freqs = np.bincount([sample["label"] for sample in train_dataset]) / len(
        train_dataset
    )
    class_freqs = torch.tensor(class_freqs, device=device, dtype=torch.float32)
    n_classes = len(class_freqs)
    n_patches = (                                # H x W / patch_size^2
        train_dataset[0]["img"].shape[-1]
        * train_dataset[0]["img"].shape[-2]
        // model_default_args["patch_size"] ** 2
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    if args.resume_training:
        checkpoint = torch.load(args.saved_model_path, map_location=device)
        model_args = checkpoint["model_args"]
        epoch_start = checkpoint["epoch"] + 1
        epoch_end = args.n_epochs + epoch_start
        best_loss_val = checkpoint["best_loss_val"]
        model = VisionTransformer(**model_args).to(device)
        state_dict = checkpoint["model_state_dict"]
        # fix the keys of the state dictionary.
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        print("[train.py]: Resuming training...")
    else:
        model_args = dict(
            n_classes=n_classes, class_freqs=class_freqs, n_patches=n_patches, **model_default_args
        )
        model = VisionTransformer(**model_args).to(device)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    print(model)
    if args.compile:
        model = torch.compile(model)
    wandb_run_name = (
        f"VisionTransformer_lr_{args.learning_rate}_batch_{args.batch_size}"
    )
    wandb.init(project=wandb_project, name=wandb_run_name, config=args)

    batches_done = 0
    print("[train.py]: Training started...")
    print(
        f"[train.py]: Total Epochs: {args.n_epochs} \t Batches per epoch: {len(train_dataloader)} "
        f"\t Total batches: {len(train_dataloader) * args.n_epochs}"
    )
    for epoch in range(epoch_start, epoch_end):
        y_preds = []
        y_train = []
        losses = []
        print(f"[train.py] Training Epoch {epoch}")
        for i, data in tqdm(enumerate(train_dataloader)):
            if batches_done == 0:
                # Log the first batch of images
                img_grid = torchvision.utils.make_grid(data["img"], nrow=16)
                wandb.log({"Images from Batch 0": wandb.Image(img_grid)})
                wandb.watch(model, loss_fn, log="all", log_freq=100)

            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            preds = F.softmax(outputs, dim=1).argmax(dim=1)
            y_preds.append(preds.cpu().numpy())
            y_train.append(labels.cpu().numpy())
            losses.append(loss.item())

            batches_done += 1

        # Train Metrics
        loss_train = torch.tensor(losses).mean().item()
        y_train, y_preds = np.concatenate(y_train), np.concatenate(y_preds)
        train_metrics, train_conf_matrix = evaluation_metrics(y_train, y_preds)

        # Validation metrics
        y_val, y_preds_val, loss_val = perform_inference(
            model, valid_dataloader, device, loss_fn
        )
        val_metrics, val_conf_matrix = evaluation_metrics(y_val, y_preds_val)

        # wandb logging
        train_metrics["Loss"], val_metrics["Loss"] = loss_train, loss_val
        wandb_log = {"epoch": epoch}
        for metric in train_metrics:
            wandb_log[f"{metric}_train"] = train_metrics[metric]
            wandb_log[f"{metric}_validation"] = val_metrics[metric]
        fig1 = conf_matrix_plot(train_conf_matrix, "Train")
        fig2 = conf_matrix_plot(val_conf_matrix, "Validation")
        wandb_log["Train Confusion Matrix"] = wandb.Image(fig1)
        wandb_log["Validation Confusion Matrix"] = wandb.Image(fig2)
        wandb.log(wandb_log)

        print(f"EPOCH: {epoch}")
        print(
            f'[TRAINING METRICS] Loss: {loss_train} | Accuracy: {train_metrics["accuracy"]} | '
            f'F1: {train_metrics["f1_score"]} | Precision: {train_metrics[f"precision"]} | Recall:'
            f'{train_metrics["recall"]}'
        )
        print(
            f'[VALIDATION METRICS] Loss: {loss_val} | Accuracy: {val_metrics["accuracy"]} | '
            f'F1: {val_metrics["f1_score"]} | Precision: {val_metrics[f"precision"]} | Recall:'
            f'{val_metrics["recall"]}'
        )

        checkpoint = {
            "epoch": epoch,
            "model_args": model_args,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "best_loss_val": best_loss_val,
        }
        print(f"[train.py]: Saving model at epoch {epoch}...")
        torch.save(
            checkpoint,
            os.path.join(args.output_dir, "saved_models", f"ckpt_epoch_{epoch}.pt"),
        )
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"[train.py]: Found new best model at epoch {epoch}. Saving model...")
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "saved_models", "0.pt"),
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
