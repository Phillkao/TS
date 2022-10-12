from torch import optim
from torch.optim import lr_scheduler

from data_preprocessing import *
from data import *
from model import *
from grad_cam import *
from utils import *

import argparse as args

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = 'Data/KN03_hourly_preprocessed_new_pipeline_20220719.csv'
ROLLING = True
MODE = 'PWR_std'

X_raw, Y_raw, X_time_step, Y_time_step, X_rolling, Y_rolling, FEATURES_NAME = data_gen(path=PATH,
                                                                                       time_step=120,
                                                                                       target_time_length=24,
                                                                                       target_diff=False,
                                                                                       add_lags=True, add_current=True)
if ROLLING:
    Y = Y_rolling
    X = X_rolling.reshape((len(X_rolling), 1, X_rolling.shape[1], X_rolling.shape[2]))
else:
    Y = Y_time_step
    X = X_time_step.reshape((len(X_time_step), 1, X_time_step.shape[1], X_time_step.shape[2]))

Y, cls_name = kmeans_label(Y, mode=MODE)

train_loader, test_loader, sample_weight = create_dataloader(X, Y, 'PWR_std_cat3', batch_size=50)
xcm = XCM(input_shape=(X.shape[1], X.shape[2]), n_class=3, filters_num=128).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(xcm.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
xcm, cls_1_xcm, history_1_xcm = train_model(xcm, criterion, optimizer, train_loader, test_loader, exp_lr_scheduler,
                                            cls_name, num_epochs=100, device=DEVICE, mode='STD_3class_120_rm_xcm',
                                            verbose=True)
