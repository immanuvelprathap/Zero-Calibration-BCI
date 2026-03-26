import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Setup a dummy model and optimizer to visualize the curve
model = torch.nn.Linear(10, 10)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

total_epochs = 300
# max_lr is usually 10x the starting lr
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                          steps_per_epoch=1, epochs=total_epochs)

lrs = []
moms = []

for epoch in range(total_epochs):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]['lr'])
    # In AdamW, momentum is tracked in 'betas'
    moms.append(optimizer.param_groups[0]['betas'][0])
    scheduler.step()

# Plotting the "Mountain" and the "Valley"
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(lrs, color='firebrick', linewidth=2)
plt.title("Learning Rate (The Mountain)")
plt.xlabel("Epoch")
plt.ylabel("LR Value")

plt.subplot(1, 2, 2)
plt.plot(moms, color='royalblue', linewidth=2)
plt.title("Momentum (The Valley)")
plt.xlabel("Epoch")
plt.ylabel("Momentum Value")

plt.tight_layout()
plt.show()