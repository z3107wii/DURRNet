import os
from os.path import join
import torch.backends.cudnn as cudnn
import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import random
import torch
import numpy as np

# 1. 解析參數
opt = TrainOptions().parse()
cudnn.benchmark = True

# 2. 資料路徑設定
base_path = "/content/drive/MyDrive/Colab Notebooks/Term_Project/training set"

# 建立訓練資料集 (4組)
train_ds1 = datasets.CEILTrainDataset(
    join(base_path, "training set 1_13700"),
    m_dir="syn",
    t_dir="t",
    enable_transforms=True,
    if_align=opt.if_align,
)
train_ds2 = datasets.CEILTrainDataset(
    join(base_path, "training set 2_Berkeley_Real"),
    m_dir="blended",
    t_dir="transmission_layer",
    enable_transforms=True,
    if_align=opt.if_align,
)
train_ds3 = datasets.CEILTrainDataset(
    join(base_path, "training set 3_Nature"),
    m_dir="blended",
    t_dir="transmission_layer",
    enable_transforms=True,
    if_align=opt.if_align,
)
train_ds4 = datasets.CEILTrainDataset(
    join(base_path, "training set 4_unaligned_train250"),
    m_dir="blended",
    t_dir="transmission_layer",
    enable_transforms=True,
    if_align=opt.if_align,
)

train_dataset_fusion = datasets.FusionDataset(
    [train_ds1, train_ds2, train_ds3, train_ds4], [0.4, 0.2, 0.2, 0.2]
)
train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=opt.nThreads,
    pin_memory=True,
)

# 驗證集
val_ds = datasets.CEILTestDataset(
    join(base_path, "training set 4_unaligned_train250"),
    m_dir="blended",
    t_dir="transmission_layer",
    enable_transforms=False,
    if_align=opt.if_align,
)
val_dataloader = datasets.DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True
)

# 3. 初始化 Engine
engine = Engine(opt)
result_dir = os.path.join(
    f"/content/drive/MyDrive/Colab Notebooks/Term_Project/DURRNet/results",
    mutils.get_formatted_time(),
)


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, "lr", lr)


# --- 訓練計數器與參數 ---
save_interval = 1500  # 每 1500 張圖片儲存一次
# 如果是接續訓練，嘗試從已訓練的 iterations 換算圖片數
accumulated_imgs = engine.iterations * opt.batchSize
current_step_counter = 0  # 用於觸發每 1500 張儲存的計數器

decay_rate = 0.5
engine.model.opt.lambda_gan = 0
set_learning_rate(opt.lr)

print(f"[i] 初始狀態：已處理圖片 {accumulated_imgs} 張")

# --- 訓練主迴圈 ---
while engine.epoch < opt.nEpochs:
    # 學習率調整策略
    if opt.fixed_lr == 0:
        if engine.epoch >= opt.nEpochs * 0.2:
            engine.model.opt.lambda_gan = 0.0001
        if engine.epoch >= opt.nEpochs * 0.4:
            set_learning_rate(opt.lr * decay_rate**1)
        if engine.epoch >= opt.nEpochs * 0.6:
            set_learning_rate(opt.lr * decay_rate**2)
        if engine.epoch >= opt.nEpochs * 0.8:
            set_learning_rate(opt.lr * decay_rate**3)
    else:
        set_learning_rate(opt.fixed_lr)

    print("\nEpoch: %d" % engine.epoch)
    engine.model.train()

    for i, data in enumerate(train_dataloader_fusion):
        engine.model.set_input(data, mode="train")
        engine.model.optimize_parameters()

        # 累計圖片張數
        accumulated_imgs += opt.batchSize
        current_step_counter += opt.batchSize

        # 顯示進度
        errors = engine.model.get_current_errors()
        util.progress_bar(
            i,
            len(train_dataloader_fusion),
            f"Total Imgs: {accumulated_imgs} | Loss: {errors}",
        )

        # 觸發定期儲存
        if current_step_counter >= save_interval:
            # 檔名範例: DURRNet_Project_iter_1500.pt
            label_name = f"iter_{accumulated_imgs}"
            print(f"\n[Checkpoint] 已達處理間隔，儲存權重: {label_name}")

            engine.model.save(label=label_name)
            engine.model.save(label="latest")  # 同時更新一個最新的

            current_step_counter = 0  # 重置區段計數器

        engine.iterations += 1

    # Epoch 結束後的驗證與儲存
    engine.epoch += 1
    save_dir = os.path.join(result_dir, "%03d" % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.eval(val_dataloader, dataset_name="val_set", savedir=save_dir, suffix="val")
    engine.model.save(label="latest")
