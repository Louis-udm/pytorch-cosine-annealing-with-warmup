# import matmlplot.pyplot as plt
import pandas as pd
import numpy as np
import torch

import cosine_annealing_warmup.scheduler as scheduler

import torch
import pandas as pd

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.AdamW(lr=1e-3, params=model.parameters())
# lr_sch = scheduler.CosineAnnealingWarmupRestarts(
#     first_cycle_steps=10,
#     cycle_mult=2.0,
#     max_lr=1e-3,
#     min_lr=2e-4,
#     warmup_steps=5,
#     gamma=0.5,
#     optimizer=optimizer,
# )

# lr_sch = scheduler.CosineAnnealingWarmupRestarts(
#     optimizer,
#     first_cycle_steps=200,
#     cycle_mult=1.0,
#     max_lr=0.1,
#     min_lr=0.05,
#     warmup_steps=50,
#     gamma=1.0,
# )

lr_sch = scheduler.CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=200,
    cycle_mult=1.0,
    max_lr=0.1,
    min_lr=0.05,
    warmup_steps=50,
    gamma=0.8,
)

# lr_sch = scheduler.CosineAnnealingWarmupRestarts(
#     optimizer,
#     first_cycle_steps=200,
#     cycle_mult=1.0,
#     max_lr=0.1,
#     min_lr=0.001,
#     warmup_steps=50,
#     gamma=0.5,
# )

# min_lr 不变，但是max_lr会根据gamma和周期变小，
# 当max_lr==min_lr时, 这个周期lr一直==min_lr
# 当max_lr<min_lr时, 这个周期lr的warmup会从min_lr下降，然后再上升到min_lr，后面的周期会降得越来越多
# lr_sch = scheduler.CosineAnnealingWarmupRestarts(
#     optimizer,
#     first_cycle_steps=200,
#     cycle_mult=1.0,
#     max_lr=0.1,
#     min_lr=0.05,
#     warmup_steps=50,
#     gamma=0.5,
# )

# lr_sch=pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
#     optimizer, T_0=10, T_mult=2, eta_min=1e-4, last_epoch=-1
# )
lrs = []
steps = []
for step in range(2000):
    if step==400:
        print(step)
    lr_sch.step()
    steps.append(step)
    lrs.append(lr_sch.get_lr()[0])
    # print(epoch, lr_sch.get_last_lr())

df = pd.DataFrame({"step": steps, "lr": lrs})
# df = pd.DataFrame({"step": steps[1100:], "lr": lrs[1100:]})
df.plot(x="step", y="lr")
print("done")
