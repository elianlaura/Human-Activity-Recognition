import torch
import numpy as np
from tqdm import tqdm


def sampling(model, diffusion, sample_n, length, feature_n, 
             batch_size, name, use_ddim=False):
    """Generate samples given the model."""
    model.eval()
    with torch.no_grad():
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        res = []
        target_n = sample_n * 2
        num_batches = (target_n + batch_size - 1) // batch_size
        print("target_n ", target_n)
        print("num_batches ", num_batches)
        print("type of sample_n ", type(sample_n))
        print("type of batch_size ", type(batch_size))
        print("sample_n ", sample_n)
        print("batch_size ", batch_size)
        for i in range(num_batches):
            res.append(
                sample_fn(
                    model,
                    (batch_size, length, feature_n),
                    clip_denoised=True,
                )
            )
            if (i + 1) % 10 == 0:
                concatenated_tensor = torch.cat(res, dim=0)[:target_n]
                np.save(name+"_"+str(i)+"_.npy",  concatenated_tensor.cpu() )
                print(f"Saved {name}.npy with {concatenated_tensor.shape[0]} samples")
    concatenated_tensor = torch.cat(res, dim=0)[:target_n]
    return concatenated_tensor  