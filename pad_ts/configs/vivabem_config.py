import argparse
import gaussian_diffusion_loss as gd
import time

# Generate the current time
ts = time.strftime("%Y%m%d-%H%M%S")

class Training_args(argparse.Namespace):
    data_dir = ""
    schedule_sampler = "batch"
    lr = 1e-4
    weight_decay = 0.0
    lr_anneal_steps = 300000   # 2 1000000
    microbatch = -1  # -1 disables microbatches
    log_interval = 100
    save_interval = 5e3
    mmd_alpha = 0.0005
    save_dir = f"./OUTPUT/vivabem_{ts}/"


class Model_args(argparse.Namespace):
    hidden_size = 256
    num_heads = 4
    n_encoder = 1
    n_decoder = 3
    feature_last = True
    mlp_ratio = 4.0
    input_shape = (24, 9)


class Diffusion_args(argparse.Namespace):
    predict_xstart = True
    diffusion_steps = 500 #10 500
    noise_schedule = "cosine"
    loss = "MSE_MMD"
    rescale_timesteps = False


class DataLoader_args(argparse.Namespace):
    batch_size = 64     # 64
    shuffle = True
    num_workers = 0
    drop_last = True            
    pin_memory = True


class Data_args(argparse.Namespace):
    name = "vivabem"
    proportion = 1.0
    data_root = "./dataset/vivabem_train_N_9_values.csv"  # "./dataset/drinkeat_5040_94u_train.csv", drinkeat_94u_train.csv
    window = 24
    save2npy = True
    neg_one_to_one = True
    seed = 123
    period = "train"
    dim = 28
