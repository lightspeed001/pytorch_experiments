import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Hyperparameters

BATCH_SIZE = 128
IMAGE_SIZE= 32
T = 1000 # Total timestamps
LEARNING_RATE = 2e-4
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#Diffusion parameters
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# U-Net Model
class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = nn.Linear(time_emb_dim, out_ch)
    if up:
      self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.bnorm1 = nn.BatchNorm2d(out_ch)
    self.bnorm2 = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU()


 def forward(self, x, t):
    # Faret convolution
    h = self.bnorm1(self.relu(self.conv1(x)))
    # Time embedding
    time_emb = self.relu(self.time_mlp(t))
    # Extend last 2 dimensions
    time_emb = time_emb[(..., ) + (None, ) * 2]
    # And time channel
    h = h + time_emb
    # Second convolution
    h =  self.bnorm2(self.relu(self.conv2(h)))
    # Down or Upsample
    return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(1000) / (half_dim - 1)
        # !!
        embeddings = torch.exp(torch.arange(half_dim, device=device) * - embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        #Time scheduling
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.con0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])

        # Upsample
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)

# Forward diffusion process
@torch.no_grad()
def forward_diffusion(x_0, t, device="cuda"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t]).reshape(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    # get data
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0,5))
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 )

    # Initialize model
    model = UNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader)
        for step, (images, _) in enumerate(pbar):
            optimizer.zero_grad()

            images = images.to(DEVICE)

            # Sample random timesteps
             t = torch.randint(0, T, (images.shape[0],), device=DEVICE).long()

             # Foward diffusion
             x_t, noise = forward_diffusion(images,t, DEVICE)

             # Loss
             loss = F.mse_loss(predicted_noise, noise)
             loss.backward()
             optimizer.step()

             pbar.set_postfix(MSE-loss.item())

             # Sampling function

             @torch.no_grad()
             def sample_timestep(x, t):
                 betas_t = betas[t]
                 sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[tsqrt_recip_alphas_t = torch.sqrt(1.0/alphas[t])])
                 sqrt_secip_alphas_t = torch.sqrt(1.0/alphas[t])

                 pred_noise = model(x, t)

                 if t == 0:
                     return pred_noise
                     else:
                     noise = torch.randn_like(x)
                     return sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(betas_t) * noise

                 @torch.no_grad()
                 def sample_plot_image():
                     img_size = IMAGE_SIZE
                     img_channels = 1

                     # Start with pure noise
                     x = torch.randn((16, img_channels, img_size, img_size), device=DEVICE)

                     for i in tqdm(reversed(range(0, T)), desc='Sampling', total=T):
                     t = torch.full((16, ), i, device=DEVICE, dtype=torch.long)
                     x = sample_timestep(x, t)

                     # Denormalize
                     x = (x.clamp(-1, 1) + 1)/2
                     x = (x * 255).type(torch.uint8)

                     return x

                     #Generate and display samples
                     samples = sample_plot_image().cpu()
                     fig, axs = plt.subplots(4, 4, figsize=(8, 8))
                     for i, ax in enumerate(axs.flatten()):
                         ax.imshow(samples[i].squeeze(), cmap='gray')
                         ax.axis('off')
                     plt.tight_layout()
                     plt.show()
                         
