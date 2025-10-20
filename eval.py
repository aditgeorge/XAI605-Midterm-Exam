# %%
# During the exam, only the validation dataset is provided to you.
# We will evaluate your model using a test dataset without noise, which is NOT provided to you.

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

StudentNumber = "0000000000"  # TODO: Replace with your student number


@torch.no_grad()
def evaluate_model(
    name,
    xt,
    yt,
    model,
    PLOT=True,
):
    device = "cpu"
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt)), batch_size=1024
    )

    pred_list = []
    mse, n = 0.0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        # pred = point_pred(model, xb, pi_index=pi_index)
        pred = model(xb)
        pred_list.append(pred.cpu().numpy())
        mse += nn.functional.mse_loss(pred, yb, reduction="sum").item()
        n += len(xb)
    mse /= n

    if PLOT:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title(f"Evaluation Result on {name} dataset")
        plt.scatter(xt, yt, color="blue", s=8, label="Test Data")
        plt.scatter(xt, np.concatenate(pred_list), color="red", s=8, label="Prediction")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()

    return {
        "name": name,
        "Error": mse,
    }


def main():
    # load each train dataset, val dataset, test dataset
    data_test = np.load(f"data1/clean_data.npz")
    xt, yt = data_test["x"], data_test["y"]

    mdn = torch.jit.load(f"model/{StudentNumber}.pth", map_location="cpu")

    res = evaluate_model(StudentNumber, xt, yt, mdn, PLOT=True)
    print(res)


if __name__ == "__main__":
    main()
