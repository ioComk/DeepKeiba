import sys
from torch.utils.data import DataLoader, random_split
import torch as t
import torch_optimizer as t_opt

# モデルのネットワーク構造をtxt形式で出力
def export_network(model, out_dir) -> None:

    with open(f'{out_dir}/network.txt', 'w') as f:
        print(f'{model}\n', file=f)

        for param in model.state_dict():
            print(f'{param}\t{model.state_dict()[param].size()}', file=f)

def generate_dataloader(dataset, batch_size, split_ratio=0.8, shuffle=[True, False], worker=0, pin_memory=False):
    '''
    [params]
        dataset: instance of Dataset class
        batch_size: mini batch size
        split_ratio: proportion of training data in the overall data set
        shuffle: whether to shuffle the dataset (list object: [train, val])
        ---------------
        [Speed-up option]
        worker: number of workers
        pin_memory: if True, the CPU memory area will not be paged
    [return]
        train_loader: dataloader for training 
        val_loader: dataloader for validation
    '''
    samples = len(dataset)
    train_size = int(samples*split_ratio)
    val_size = samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle[0], num_workers=worker, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=shuffle[1], num_workers=worker, pin_memory=pin_memory)

    return train_loader, val_loader

def check_loss(epoch, itr, interval, running_loss, logger=None):
        print(f'[{epoch+1}, {itr+1:03}] loss: {running_loss/interval:.5f}')

        if logger != None:
            logger.log({'Training loss': running_loss/interval})

def get_optimzer(optimizer_name, model, lr):

    if optimizer_name == 'Adam':
        optimizer = t.optim.Adam(model.parameters(), lr)
    elif optimizer_name == 'RAdam':
        optimizer = t_opt.RAdam(model.parameters(), lr)
    elif optimizer_name == 'SGD':
        optimizer = t.optim.SGD(model.parameters(), lr)
    else:
        print(f'{optimizer_name} is not supported.')
        sys.exit(0)

    return optimizer