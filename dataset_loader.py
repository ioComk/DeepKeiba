import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class TrainValDataset(Dataset):

    def __init__(self, path):

        self.dataset_path = Path(path)

        self.df = []
        for src in self.dataset_path.iterdir():
            self.df.append(pd.read_csv(str(src), encoding='shift-jis'))

    def __len__(self):
        return len(list(self.dataset_path.iterdir()))
    
    def __getitem__(self, i):
        
        label = self.df[i][['着順', '馬番']]
        label = label.sort_values('着順').to_numpy().astype(np.float32)

        # 1~18位を0~17に
        label = label[0, 1] - 1

        # One-hot encoding

        drop_col = ['着順', '馬番', '人気']
        
        df_np = self.df[i].drop(drop_col, axis=1).to_numpy().astype(np.float32)
        input_vec = df_np.flatten()

        # 最大出走数18頭なので，ベクトルサイズを揃える
        if len(input_vec) < 270:
            zero_pad = np.zeros(270-len(input_vec), dtype='float32')
            input_vec = np.concatenate([input_vec, zero_pad])

        return input_vec, label