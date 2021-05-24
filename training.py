import numpy as np
import torch as t
from torch import nn
import os, sys
from pathlib import Path
import optuna
import wandb

from dataset_loader import TrainValDataset
from net import Model
from earlystopping import EarlyStopping
from functions import export_network, generate_dataloader, check_loss, get_optimzer

os.environ['WANDB_SILENT'] = "true"

MAX_EPOCH = 500
PATIENCE  = 20

# Training
def train_step(epochs, data_loader, model, optimizer, run):

    running_loss = 0.0
    check_interval = 5

    model.train()

    for i, data in enumerate(data_loader):

        df, label = [i.to(device) for i in data]

        label = label.long()
        # initialize optimizer
        optimizer.zero_grad()

        output = model(df)

        # loss計算
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # check_interval毎にTraining lossを表示(ログに追加)
        if (i+1) % check_interval == 0:
            check_loss(epochs, i, check_interval, running_loss, run)
            running_loss = 0.0
        
# Validation
def val_step(model, data_loader, early_stopping):

    acc_sum = 0.0
    model.eval()

    with t.no_grad():
        for i, data in enumerate(data_loader):
            
            df, label = [i.to(device) for i in data]
            label = label.long()

            output = model(df)
            _, predicted = t.max(output.data, 1)

            acc_sum += (predicted == label).sum().item() / len(label)

        acc = acc_sum / (i+1)
        early_stopping(acc, model)

        print('Accuracy: ', acc)

    return acc

def objective(trial):

    # ハイパーパラメータ
    batch_size = trial.suggest_int('batch_size', 16, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'RAdam', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-2)

    config = dict(
        batch = batch_size,
        dropout = dropout,
        optimizer = optimizer_name,
        learning_rate = lr,
        best_val_score = None,
    )

    # WandB setting
    run = wandb.init(project='keiba', config=config, reinit=True)

    # モデル等を出力するフォルダを設定
    out_dir = wandb.run.dir
    path = Path(out_dir)
    out_dir = '/'.join(path.parts[0:-1])

    # Dataloader準備
    dataset_path = './dataset'
    dataset = TrainValDataset(dataset_path)
    train_loader, val_loader = generate_dataloader(dataset, batch_size)

    # DNNモデル定義
    model = Model(dropout).to(device)

    optimizer = get_optimzer(optimizer_name, model, lr)

    val_loss = 0
    best_val_score = np.Inf

    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir=out_dir)

    #####################    DNN Training     #########################
    for epoch in range(MAX_EPOCH):

        train_step(epoch, train_loader, model, optimizer, run)
        val_loss = val_step(model, val_loader, early_stopping)

        # update log
        run.log({'Validation loss': val_loss, 'Epoch': epoch+1})
        run.config.update({'best_val_score': val_loss if val_loss < best_val_score else best_val_score}, allow_val_change=True)

        trial.report(val_loss, epoch)

        # Earlystopping
        if early_stopping.early_stop:
            print('Early stopping.')
            break
        
        # # 枝刈り
        if trial.should_prune():
            run.config.update({'PRUNED': True}, allow_val_change=True)
            export_network(model, wandb.run.dir)
            raise optuna.exceptions.TrialPruned()
    ###################################################################

    export_network(model, wandb.run.dir)
    run.finish()  # Close WandB logger

    return val_loss

if __name__ == "__main__":

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print('Device:', device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # 枝刈り手法
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials= 10,
        n_warmup_steps= 10,
        interval_steps= 1
    )

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100)
