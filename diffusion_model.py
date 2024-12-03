'''
ACKNOWLEDGEMENT: THE CODE IS PARTLY BASED ON STABLE DIFFUSION (https://github.com/Stability-AI/stablediffusion) AND PIXELCNN (https://github.com/ermongroup/ddim/blob/main/models/diffusion.py)
'''
import copy
from scipy.io import loadmat
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
from tqdm import tqdm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from PIL import Image
import os
import math


def set_device():
    """
    Set the device. CUDA if available, CPU otherwise
    
    Args:
    None

    Returns:
    Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook. \n"
            "If you want to enable it, in the menu under `Runtime` -> \n"
            "`Hardware accelerator.` and select `GPU` from the dropdown menu")
    else:
        print("GPU is enabled in this notebook. \n"
            "If you want to disable it, in the menu under `Runtime` -> \n"
            "`Hardware accelerator.` and select `None` from the dropdown menu")

    return device


def get_movie(movie_name, dim):
    '''Prepare the movie frames.'''
    # boc = BrainObservatoryCache()
    # session_id = boc.get_ophys_experiments(experiment_container_ids=[511510736], stimuli=[movie_name])[0]['id']
    # data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
    # natural_movie_template = data_set.get_stimulus_template(movie_name)

    duration = 30
    fps = 30

    # if not os.path.isdir('movie_frame'):
    #     os.mkdir('movie_frame')

    # for i in range(int(duration*fps)):
    #     frame = natural_movie_template[i]
    #     cv2.imwrite(f"./movie_frame/frame_{i}.jpg", frame)
        
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # To (0,1)
        torchvision.transforms.Resize(dim),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # To (-1,1)
    ])
        
    data_set = torch.zeros(1, 1, dim[0], dim[1])
    for i in range(int(duration*fps)):
        image = Image.open(f"./movie_frame/frame_{i}.jpg")
        
        transformed_image = transform(image)
        data_set = torch.cat((data_set, transformed_image.unsqueeze(1)), dim=0)
    data_set = data_set[1:]
    return data_set


def prepare_neural_data(movie_data_set, mouse_id, pre_time, post_time, days, bs=0.25, train=False, ratio_drop_train=0):
    '''Prepare the neural data that corresponds to each movie frame'''
    pad_to = int(pre_time/bs) + int(post_time/bs)

    data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/baselineOriginalLatentState0_trans.mat")['baselineOriginalLatentState_mean_all']

    num_neuron = data_np_all_movie.shape[0]
    X_np = np.empty((len(days)*int(120/pad_to)*20, pad_to, num_neuron)) # 20 repeats
    y = np.empty((len(days)*int(120/pad_to)*20, 1, movie_data_set.shape[2], movie_data_set.shape[3]))

    i = 0
    for day_idx_idx, day_idx in tqdm(enumerate(days), desc='day'):
        if day_idx == 0:
            data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/baselineOriginalLatentState0_trans.mat")['baselineOriginalLatentState_mean_all']
        else:
            data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/updatedInstabilitiesLatentState{day_idx}_trans.mat")['updatedInstabilitiesLatentState_mean_all']
        
        for repeat in range(data_np_all_movie.shape[1]):
            
            frame_per_bin = 7.5 # 900/120
            repeat_data = data_np_all_movie[:, repeat, :]
            for first_frame_idx in range(data_np_all_movie.shape[2] - pad_to + 1):
                if first_frame_idx % pad_to != 0:
                    continue
                sample = repeat_data[:, first_frame_idx: first_frame_idx+pad_to]
                if train:
                    if torch.rand(1).item() < ratio_drop_train:
                        sample = torch.zeros(sample.shape)
                X_np[i] = sample.T
                sample_y = movie_data_set[int(np.ceil(first_frame_idx * frame_per_bin + int(15*pad_to/6)))].numpy()
                y[i] = sample_y
                i += 1
    X = torch.tensor(X_np)
    y = torch.tensor(y)

    return X, y
    
    
def prepare_neural_data_one_sample(movie_data_set, mouse_id, pre_time, post_time, days, bs=0.25, train=False, ratio_drop_train=0, seed=42):
    '''Prepare the data (same (x_0, y) pairs) for distribution visualization'''
    pad_to = int(pre_time/bs) + int(post_time/bs)

    data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/baselineOriginalLatentState0_trans.mat")['baselineOriginalLatentState_mean_all']

    num_neuron = data_np_all_movie.shape[0]
    X_np = np.empty((len(days)*int(120/pad_to)*20, pad_to, num_neuron)) # 20 repeats
    y = np.empty((len(days)*int(120/pad_to)*20, 1, movie_data_set.shape[2], movie_data_set.shape[3]))
    
    np.random.seed(seed)
    day_idx_choose = np.random.choice(days)
    if day_idx_choose == 0:
            data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/baselineOriginalLatentState0_trans.mat")['baselineOriginalLatentState_mean_all']
    else:
        data_np_all_movie = loadmat(f"./my_data/Mouse{mouse_id}/updatedInstabilitiesLatentState{day_idx_choose}_trans.mat")['updatedInstabilitiesLatentState_mean_all']
    repeat_choose = np.random.choice(range(data_np_all_movie.shape[1]))
    frame_per_bin = 7.5 # 900/120    
    repeat_data = data_np_all_movie[:, repeat_choose, :]
    window_choose = np.random.choice(range(10))
    
    first_frame_idx = window_choose*pad_to
    sample = repeat_data[:, first_frame_idx: first_frame_idx+pad_to]
    
    sample_y = movie_data_set[int(np.ceil(first_frame_idx * frame_per_bin + int(15*pad_to/6)))].numpy()
    
    for i in range(len(days)*int(120/pad_to)*20):
    
        X_np[i] = sample.T
        y[i] = sample_y
    
    X = torch.tensor(X_np)
    y = torch.tensor(y)

    return X, y
    

def show_grid(imgs, title=""):
    '''Visualization tool'''
    fig, ax = plt.subplots(figsize=(20,20))
    imgs = [ (img - img.min()) / (img.max() - img.min()) for img in imgs ] # Normalize to [0, 1] for imshow()
    img = torchvision.utils.make_grid(imgs, padding=1, pad_value=1).numpy()
    ax.set_title(title)
    ax.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
    ax.set(xticks=[], yticks=[])
    plt.show()


def inference_ddim_deterministic(model, val_loader, T, alphas, alpha_bars, sigmas,  H, W, w=0.8, quick_probe=True):
    '''Generate images from ddim deterministic diffusion model.'''
    model.eval()
    # torch.manual_seed(42)
    with torch.no_grad():
        n_channels: int = 1 # 1 for grayscale
        # x_T \sim N(0, I)
        loss_all = []
        generated = []
        original = []
        for nd, image in tqdm(val_loader, desc='batch'):
        
            x_T = torch.randn((nd.shape[0], n_channels, H, W))
            # For t = T ... 1
            x_t = x_T
            x_ts = [] # save image as diffusion occurs
            x_ts.append(x_t)
            for t in range(T-1, -1, -1):
    
                t_vector = torch.fill(torch.zeros((nd.shape[0],)), t)
                epsilon_theta = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd.type(torch.float32).to(model.device)).to('cpu')
                nd_zero = torch.zeros(nd.shape)
                epsilon_theta_zero = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd_zero.type(torch.float32).to(model.device)).to('cpu')
                epsilon_theta = (w+1) * epsilon_theta - w * epsilon_theta_zero
                # Setup terms for x_t-1
                
                if t > 0:
                    x_t_1 = torch.sqrt(alpha_bars[t-1]) * (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta) / torch.sqrt(alpha_bars[t]) + torch.sqrt(1-alpha_bars[t-1])*epsilon_theta
                else:
                    z = torch.randn(x_t.shape)
                    x_t_1 = (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta)/ torch.sqrt(alpha_bars[t])  + sigmas[0]*z
                x_t = x_t_1
                x_ts.append(x_t)
                
            loss: float = torch.sum((image - x_ts[-1])**2)/nd.shape[0]
            loss_all.append(loss.item())
            original.append(image)
            generated.append(x_t)
            if quick_probe:
                break
        return torch.stack(x_ts).transpose(0, 1), np.mean(loss_all), generated, original
    

def inference_ddim_stochastic(model, val_loader, T, alphas, alpha_bars, sigmas,  H, W, w=0.8, quick_probe=True):
    '''Generate images from ddim stochastic diffusion model.'''
    model.eval()
    # torch.manual_seed(42)
    with torch.no_grad():
        # Dimensions
        n_channels: int = 1 # 1 for grayscale
        # x_T \sim N(0, I)
        loss_all = []
        generated = []
        original = []
        for nd, image in tqdm(val_loader, desc='batch'):
        
            x_T = torch.randn((nd.shape[0], n_channels, H, W))
            # For t = T ... 1
            x_t = x_T
            x_ts = [] # save image as diffusion occurs
            x_ts.append(x_t)
            for t in range(T-1, -1, -1):
                z = torch.randn(x_t.shape)
                # Setup terms for x_t-1
                t_vector = torch.fill(torch.zeros((nd.shape[0],)), t)
                
                epsilon_theta = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd.type(torch.float32).to(model.device)).to('cpu')
                nd_zero = torch.zeros(nd.shape)
                epsilon_theta_zero = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd_zero.type(torch.float32).to(model.device)).to('cpu')
                epsilon_theta = (w+1) * epsilon_theta - w * epsilon_theta_zero
                if t > 0:
                    x_t_1 = torch.sqrt(alpha_bars[t-1]) * (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta) / torch.sqrt(alpha_bars[t]) + torch.sqrt(1-alpha_bars[t-1]-sigmas[t]**2)*epsilon_theta + sigmas[t]*z
                else:
                    x_t_1 = (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta)/ torch.sqrt(alpha_bars[t])  + sigmas[t]*z

                x_t = x_t_1
                x_ts.append(x_t)
                
            loss: float = torch.sum((image - x_ts[-1])**2)/nd.shape[0]
            loss_all.append(loss.item())
            original.append(image)
            generated.append(x_t)
            if quick_probe:
                break
        return torch.stack(x_ts).transpose(0, 1), np.mean(loss_all), generated, original
    

def inference_ddim_deterministic_speedup(model, val_loader, T, alphas, alpha_bars, sigmas,  H, W, w=0.8, quick_probe=True, keep_every=3):
    '''Generate images from ddim deterministic (spedup) diffusion model.'''
    model.eval()
    # torch.manual_seed(42)
    t_range = list(range(T-1, -1, -keep_every))
    if t_range[-1] != 0:
        t_range.append(0)
    with torch.no_grad():
        # Dimensions
        n_channels: int = 1 # 1 for grayscale
        # x_T \sim N(0, I)
        loss_all = []
        generated = []
        original = []
        for nd, image in tqdm(val_loader, desc='batch'):
        
            x_T = torch.randn((nd.shape[0], n_channels, H, W))
            # For t = T ... 1
            x_t = x_T
            x_ts = [] # save image as diffusion occurs
            x_ts.append(x_t)
            for t_i, t in enumerate(t_range):
                
                z = torch.randn(x_t.shape)
                # Setup terms for x_t-1
                t_vector = torch.fill(torch.zeros((nd.shape[0],)), t)
                
                epsilon_theta = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd.type(torch.float32).to(model.device)).to('cpu')
                nd_zero = torch.zeros(nd.shape)
                epsilon_theta_zero = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd_zero.type(torch.float32).to(model.device)).to('cpu')
                epsilon_theta = (w+1) * epsilon_theta - w * epsilon_theta_zero
                if t > 0:
                    t_prev = t_range[t_i+1]
                    x_t_1 = torch.sqrt(alpha_bars[t_prev]) * (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta) / torch.sqrt(alpha_bars[t]) + torch.sqrt(1-alpha_bars[t_prev])*epsilon_theta
                else:
                    x_t_1 = (x_t - torch.sqrt(1-alpha_bars[t])*epsilon_theta)/ torch.sqrt(alpha_bars[t])  + sigmas[0]*z

                x_t = x_t_1
                x_ts.append(x_t)
                
            loss: float = torch.sum((image - x_ts[-1])**2)/nd.shape[0]
            loss_all.append(loss.item())
            original.append(image)
            generated.append(x_t)
            if quick_probe:
                break
        return torch.stack(x_ts).transpose(0, 1), np.mean(loss_all), generated, original
    
    
def inference_ddpm(model, val_loader, T, alphas, alpha_bars, sigmas,  H, W, w=0.8, quick_probe=True, sigmat=0.1):
    '''Generate images from ddpm diffusion model.'''
    model.eval()
    # torch.manual_seed(42)
    with torch.no_grad():
        # Dimensions
        n_channels: int = 1 # 1 for grayscale
        # x_T \sim N(0, I)
        loss_all = []
        generated = []
        original = []
        for nd, image in tqdm(val_loader, desc='batch'):
        
            x_T = torch.randn((nd.shape[0], n_channels, H, W))
            # For t = T ... 1
            x_t = x_T
            x_ts = [] # save image as diffusion occurs
            x_ts.append(x_t)
            for t in range(T-1, 0, -1):
                # z \sim N(0, I) if t > 1 else z = 0
                z = torch.randn(x_t.shape) if t > 1 else torch.zeros_like(x_t)
                # Setup terms for x_t-1
                t_vector = torch.fill(torch.zeros((nd.shape[0],)), t)
                
                epsilon_theta = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd.type(torch.float32).to(model.device)).to('cpu')
                nd_zero = torch.zeros(nd.shape)
                epsilon_theta_zero = model(x_t.type(torch.float32).to(model.device), t_vector.type(torch.float32).to(model.device), nd_zero.type(torch.float32).to(model.device)).to('cpu')
                epsilon_theta = (w+1) * epsilon_theta - w * epsilon_theta_zero
                x_t_1  = (
                    1 / torch.sqrt(alphas[t]) * (x_t - (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t]) * epsilon_theta)
                    + sigmas[t] * z
                )                
                x_t = x_t_1
                x_ts.append(x_t)
                
            loss: float = torch.sum((image - x_ts[-1])**2)/nd.shape[0]
            loss_all.append(loss.item())
            original.append(image)
            generated.append(x_t)
            if quick_probe:
                break
        return torch.stack(x_ts).transpose(0, 1), np.mean(loss_all), generated, original


def train(train_dataloader, val_loader, model, T, alpha_bars, alphas, sigmas, H, W, optimizer, n_epochs, logging_steps, inference_type):
    '''Train the UNet noise predicting model, evaluated by specified inference methods'''
    torch.manual_seed(42)
    losses = []
    test_loss = []
    minimum_loss = float('Inf')
    best_model = None
    model.train()
    for _ in range(n_epochs):
        train_loss: float = 0.0
        loss_epoch = []
        # start_time = time.time()
        for batch_idx, (nd, x_0) in enumerate(train_dataloader):
            B: int = x_0.shape[0] # batch size
            # Sample x_0 ~ q(x_0)
            x_0 = x_0.to(model.device)
            # Sample t ~ U(0, T)
            t = torch.randint(0, T, (B,))
            # Sample e ~ N(0, I)
            epsilon = torch.randn(x_0.shape, device=model.device)
            # Sample x_t ~ q(x_t | x_0) = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) e
            x_0_coef = torch.sqrt(alpha_bars[t]).reshape(-1, 1, 1, 1).to(model.device)
            epsilon_coef = torch.sqrt(1 - alpha_bars[t]).reshape(-1, 1, 1, 1).to(model.device)
            x_t = x_0_coef * x_0 + epsilon_coef * epsilon
            # Predict epsilon_theta = f(x_t, t)
            epsilon_theta = model(x_t.type(torch.float32), t.type(torch.float32).to(model.device), nd.type(torch.float32).to(model.device))
            # Calculate loss
            loss: float = torch.sum((epsilon - epsilon_theta)**2)
            # Backprop gradient
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            loss_epoch.append(loss.item())
            train_loss += loss.item()
            if (batch_idx + 1) % logging_steps == 0 : 
                deno = logging_steps * B
                print(f'Train loss over last {B} batches: {train_loss/deno}')
                train_loss = 0.0

        losses.append(np.mean(loss_epoch))
        
        if inference_type == 'ddpm':
            images_test, test_loss_epoch, _, _ = inference_ddpm(model, val_loader, T, alphas, alpha_bars, sigmas, H, W)
        elif inference_type == 'ddim_stochastic':
            images_test, test_loss_epoch, _, _ = inference_ddim_stochastic(model, val_loader, T, alphas, alpha_bars, sigmas, H, W)
        else:
            images_test, test_loss_epoch, _, _ = inference_ddim_deterministic(model, val_loader, T, alphas, alpha_bars, sigmas, H, W)
        print(f'Test loss at this epoch {test_loss_epoch}')
        if test_loss_epoch <= minimum_loss:
            minimum_loss = test_loss_epoch
            best_model = copy.deepcopy(model)
        
        test_loss.append(test_loss_epoch)
            
    return best_model, losses, test_loss, images_test


def get_timestep_embedding(timesteps, embedding_dim):
    '''Sinusoidal time embedding'''
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb # (Batch, Time Embedding Size)


def nonlinearity(x):
    '''nonlinearity function'''
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    '''Group normalization'''
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    '''Upsampling block'''
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        torch.manual_seed(42)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    '''Downsampling block'''
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:

            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
        torch.manual_seed(42)

    def forward(self, x):
        if self.with_conv:            
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
    
class ResnetBlock(nn.Module):
    '''ResNET block'''
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                        out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
        torch.manual_seed(42)
                
                
    def forward(self, x, temb):
        
        # Deal with h
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None] 

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # deal with x
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    '''QKV attention mechanism block'''
    def __init__(self, in_channels, nd_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        
        # Convert neural data channels to in_channels
        self.nd_proj = torch.nn.Linear(nd_channels,
                                        in_channels)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        torch.manual_seed(42)

    def forward(self, x, nd): #nd (batch, sequence, channel/dim)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        
        nd2 = nd
        
        k = self.nd_proj(nonlinearity(nd)) #(batch, sequence, channel/dim)
        k = k.permute(0, 2, 1) # batch, channel/dim, sequence
        v = self.nd_proj(nonlinearity(nd2)) #(batch, sequence, channel/dim)
        v = v.permute(0, 2, 1) # batch, channel/dim, sequence

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        w_ = torch.bmm(q, k)     # b,hw,sequence    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)   # b,sequence,hw
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_) # batch, channel/dim, hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
    
    
class Model(nn.Module):
    '''UNet Model (Combining previous components)'''
    def __init__(self, config):
        super().__init__()
        out_ch, ch_mult = config['model']['out_ch'], tuple(config['model']['ch_mult'])
        attn_resolutions = config['model']['attn_resolutions']
        dropout = config['model']['dropout']
        resamp_with_conv = config['model']['resamp_with_conv']

        self.nd_channel = config['model']['nd_channel']
        self.ch = config['model']['ch']
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = config['model']['num_res_blocks']
        self.resolution = config['data']['image_size']
        self.in_channels = config['model']['in_channels']
        self.device = config['model']['device']

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([ # this is adding a member variable  to self.temb
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                        self.ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch*in_ch_mult[i_level] # this initially is ch (output from conv_in)
            block_out = self.ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout))
                block_in = block_out # channel count are kept same within a level
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, self.nd_channel))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1: #if not the last level, we will dosample
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2 # floor division
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in, self.nd_channel)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch*ch_mult[i_level]
            skip_in = self.ch*ch_mult[i_level] #=output channels from corresponding i-level during downsampling
            # one extra block than during downsampling period
            # deal with one extra block. this is to deal with resulting h from downsample layer
            for i_block in range(self.num_res_blocks+1): 
                if i_block == self.num_res_blocks: 
                    skip_in = self.ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, self.nd_channel))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        torch.manual_seed(42)

    def forward(self, x, t, nd):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, nd)
                hs.append(h) # h at every single block will be used during upsamling
            if i_level != self.num_resolutions-1:
                # result of downsample is also appended needed to poped and dealt with
                hs.append(self.down[i_level].downsample(hs[-1])) 

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h, nd)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, nd)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h