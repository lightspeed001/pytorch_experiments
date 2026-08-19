# Experimental Deep Learning with PyTorch
## Experimental Deep Learning Samples with PyTorch

### Included Samples: :robot:
> DDPM (Diffussion Model) - Full Denoising Diffusion Probabilistic Model for MNIST generation with U-Net, complete forward/reverse diffussion process, and sampling code. [ddpm.py](ddpm.py)
> Vision Transformer (VIT) - From-scratch implementation with patch embeddings, positional embeddings, multi-head attention, and transformer blocks. [vit.py](vit.py)
> Conditional GAN (cGAN)- MNIST generation conditioned on close labels with sperate generator/discriminator networks. [cgan.py](cgan.py)
> Neural ODE - Continuous-depth model using ODE solvers ( ``` torchdiffeq ```) with the ability to evaluate at any depth. [neural.py](neural.py)

### How to run :cloud:
  > DDPM (Difussion Model)
  
  ```bash
        pip install torch torchvision matplotlib tqdm numpy
        python ddpm_mnist.py  
  ```
  * Expected Output: After ~20 epochs on MNIST, you'll see generated digits that resembe handwritten numbers.
  > Vision Transformer (VIT)

  

