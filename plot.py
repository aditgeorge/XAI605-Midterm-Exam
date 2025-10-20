# %%
import numpy as np


def main():
    # load each train dataset, val dataset, test dataset
    data = np.load(f"data2/noisy_data.npz")
    xt, yt = data["x"], data["y"]

    # plot the data
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Clean Data")
    plt.scatter(xt, yt, color="blue", s=8, label="Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
