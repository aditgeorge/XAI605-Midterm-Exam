import torch
from torch import nn
from mdn import get_argmax_mu

class JitWrapper(torch.nn.Module):
    """
    This wrapper defines the *inference* logic for JIT compilation.
    It takes the raw model and adds the get_argmax_mu logic.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # This is your custom function for inference
        # 1. Get the GMM parameters from the base model
        pi, mu, sigma = self.model(x)
        
        # 2. Get the most probable mean (your final prediction)
        #    This uses the globally defined 'get_argmax_mu' function
        argmax_mu = get_argmax_mu(pi, mu)
        
        # 3. Return only the final prediction
        return argmax_mu

def save_jit_model(model, path):
    """Wraps, scripts, and saves the model for JIT."""
    print(f"Wrapping model for JIT scripting...")
    # Wrap the model with JitWrapper
    jit_model = JitWrapper(model)
    
    # Set to evaluation mode (important before scripting)
    jit_model.eval() 
    
    # Script the wrapped model
    scripted_mdn = torch.jit.script(jit_model)
    
    # Save the scripted model
    scripted_mdn.save(path)
    print(f"JIT-scripted model saved successfully to {path}")