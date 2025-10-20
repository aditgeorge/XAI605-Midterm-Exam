# %%
import torch
from torch import nn

StudentNumber = "0000000000"  # TODO: Replace with your student number


class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, K=2, hidden=64):
        super().__init__()
        """
        Mixture Density Network (MDN) for modeling a mixture of Gaussian distributions.
        Args:
            in_dim (int): Dimension of input features.
            out_dim (int): Dimension of output features.
            K (int): Number of mixture components.
            hidden (int): Number of hidden units in the hidden layer.

        Declare layers:
            self.net: Base neural network to extract features from input
            self.pi: Layer to output mixture weights
            self.mu: Layer to output means of Gaussian components
            self.sigma: Layer to output standard deviations of Gaussian components        
        """
        self.net = nn.Identity()  # Simple linear layer as placeholder

    def forward(self, x):
        """
        pi: Probabilities for each mixture component, shape (B, K)
        mu: Means of each mixture component, shape (B, K, out_dim)
        sigma: Standard deviations of each mixture component, shape (B, K, out_dim)

        return pi, mu, sigma
        """
        return self.net(x)


class JitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Write your custom function for inference
        return self.model(x) + 1.0  # Example modification


def save_jit_model(model, path):
    # Wrap the model with JitWrapper
    jit_model = JitWrapper(model)
    scripted_mdn = torch.jit.script(jit_model)
    scripted_mdn.save(path)


if __name__ == "__main__":
    # Example usage
    mdn = MixtureDensityNetwork(in_dim=1, out_dim=1, K=3, hidden=64)
    save_jit_model(mdn, f"model/{StudentNumber}.pth")

    # Load the JIT model
    loaded_model = torch.jit.load(f"model/{StudentNumber}.pth")
    x = torch.arange(5).float().unsqueeze(-1)
    output = loaded_model(x)
    print(output)  # Should print input + 1.0
