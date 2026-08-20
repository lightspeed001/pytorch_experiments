# Experimental Deep Learning with PyTorch
## Experimental Deep Learning Samples with PyTorch

### Included Samples: :robot:
> * __DDPM (Diffussion Model)__ - Full Denoising Diffusion Probabilistic Model for MNIST generation with U-Net, complete forward/reverse diffussion process, and sampling code. [ddpm.py](ddpm.py)  
> * __Vision Transformer (VIT)__ - From-scratch implementation with patch embeddings, positional embeddings, multi-head attention, and transformer blocks. [vit.py](vit.py)  
> * __Conditional GAN (cGAN)__- MNIST generation conditioned on close labels with sperate generator/discriminator networks. [cgan.py](cgan.py)  
> * __Neural ODE__ - Continuous-depth model using ODE solvers ( ``` torchdiffeq ```) with the ability to evaluate at any depth. [neural.py](neural.py) 

### Key Components :robot:  
#### Denoising Diffusion Probablistic Model (DDPM)  
> * __Concept__ - Gradually denoise Gaussian noise into data through a learned reversed process.  
> * __Forward Process__ - Gradually add noise to data (fixed schedule)
> * __Reverse Process__ - Learn to denoise using a U-Net
> * __Loss__ - Mean squared error between predicted and actual noise.  

#### Vision Transformer (ViT)  
> * __Concept__ - Apply transformer architectures to image patches for classification.  
> * __Patch Embedding__ - Split image into fixed-size patches  
> * __Position Embedding__ - Add positional information to patches  
> * __Transformer Encoder__ - Standard transformer blocks  
> * __Classification Head__ - MLP for final prediction  

### How to run :cloud:
  > * DDPM (Difussion Model)
  
  ```bash
        pip install torch torchvision matplotlib tqdm numpy
        python ddpm_mnist.py  
  ```
  * Expected Output: After ~20 epochs on MNIST, you should see generated digits that resembe handwritten numbers.  
  > * Vision Transformer (VIT)  

  ```bash
        
  ```

  

