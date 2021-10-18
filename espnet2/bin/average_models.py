import torch
import os
avg = None
paths = [path for path in os.listdir(".") if path.endswith("epoch.pth")]
for path in paths:
    states =  torch.load(path, map_location="cpu")

    if avg is None:
        avg = states
    else:
        # Accumulated
        for k in avg:
            avg[k] = avg[k] + states[k]

for k in avg:
    if str(avg[k].dtype).startswith("torch.int"):
        # For int type, not averaged, but only accumulated.
        # e.g. BatchNorm.num_batches_tracked
        # (If there are any cases that requires averaging
        #  or the other reducing method, e.g. max/min, for integer type,
        #  please report.)
        pass
    else:
        avg[k] = avg[k] / len(paths)

# 2.b. Save the ave model and create a symlink
torch.save(avg, "valid.acc.ave_10best.pth")
torch.save(avg, "valid.acc.ave.pth")
