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

device = "cuda" if torch.cuda.is_available() else "cpu"
model_default_args = dict(  # default arguments for the model (Paramaters)
    n_blocks=2,
    dropout=0.3,
)
wandb_project = "MyProject"


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_args_parser():
    parser = argparse.ArgumentParser("Training and evaluation script", add_help=False) 
    parser.add_argument("--batch_size", default=16, type=int) #runs from comand line? 
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--seed", default=1337, type=int) #same as seed defined before?
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--n_cpu", default=8, type=int)
    parser.add_argument("--input_dir", default="./data/", type=str)
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
    return fig #creating heat map 


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


def init_weights(module):
    """
    Initialise weights of given module using Kaiming Normal initialisation for linear and
    convolutional layers, and zeros for bias.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BasicCNN(nn.Module):
    """
    Simple CNN model
    """

    def __init__(self, n_classes, class_freqs, n_blocks, dropout) -> None: #can define n_blocks, dropout directly here, but it helps to hold them as global parameters 
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.maxpool = nn.MaxPool2d(2, 1, 1)
        self.features = self.make_feature_layers(n_blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self.apply(init_weights) #applies bias 
        self.classifier[-1].bias = nn.Parameter(torch.log(class_freqs))

    def make_feature_layers(self, n_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(n_blocks):
            layers += [nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.features(x)
        x = self.avgpool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


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


def main(args) -> None: #same as training loop 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed) #setting the seed means you reproduce the same stuff everytime you run the script 
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #creates unique id by having a date and time. This gets used to create the folder  
    args.output_dir = os.path.join(args.output_dir, time) #UUID? something that python automatically creates, its a Unique ID but it has no interpretation
    os.makedirs(os.path.join(args.output_dir, "saved_models"), exist_ok=True) #os.path.join creates based on iff you are on mac or linux or windows 

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

    epoch_start, epoch_end = 0, args.n_epochs #stored this way so we start at 0 and then up to the number of epochs 
    best_loss_val = float("inf")
    class_freqs = np.bincount([sample["label"] for sample in train_dataset]) / len(
        train_dataset
    )
    class_freqs = torch.tensor(class_freqs, device=device, dtype=torch.float32)
    n_classes = len(class_freqs)
    loss_fn = nn.CrossEntropyLoss().to(device)

    if args.resume_training: #starts from the epoch the model was last left at 
        checkpoint = torch.load(args.saved_model_path, map_location=device)
        model_args = checkpoint["model_args"]
        epoch_start = checkpoint["epoch"] + 1
        epoch_end = args.n_epochs + epoch_start
        best_loss_val = checkpoint["best_loss_val"]
        model = BasicCNN(**model_args).to(device)
        state_dict = checkpoint["model_state_dict"] #need to save
        # fix the keys of the state dictionary.
        unwanted_prefix = "_orig_mod." #this is a pytorch thing, need to account for it 
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"]) #need to save 
        print("[train.py]: Resuming training...")
    else: #starts the model from begining 
        model_args = dict(
            n_classes=n_classes, class_freqs=class_freqs, **model_default_args #makes bigger dictonary? 
        )
        model = BasicCNN(**model_args).to(device)
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    print(model)
    model = torch.compile(model)
    wandb_run_name = f"BasicCNN_lr_{args.learning_rate}_batch_{args.batch_size}"
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
            os.path.join(args.output_dir, "saved_models", f"ckpt_epoch_{epoch}.pt"), #saved for one for each epoch  #may be some case where we want less than best model
        )
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"[train.py]: Found new best model at epoch {epoch}. Saving model...")
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "saved_models", "best_model.pt"),
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
# can run on compand path when using conda environment 

# If you work in ML/AI enough you run into a case where test set has high accuracy but test and validation set has low accuracy 

#one thing you can do is look at the neual network weights and biases . There is a tool called weights and biases online that can help you see these 
