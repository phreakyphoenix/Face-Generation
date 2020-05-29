# Face Generation
### Project 4 in Udacity Deep Learning Nanodegree

In this project, you'll define and train a DCGAN on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible! 

To check out the Project, use any of these links:
1. [Markdown](https://github.com/phreakyphoenix/Face-generAtion/tree/master/README.md) 
2. [IPYNB](https://github.com/phreakyphoenix/Face-generAtion/blob/master/dlnd_face_generation.ipynb)
3. [HTML](https://github.com/phreakyphoenix/Face-generAtion/blob/master/dlnd_face_generation.html)  

>**g.pt and d.pt are the trained generator and discriminator files**

The project will be broken down into a series of tasks from **loading in data to defining and training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.

### Get the Data

You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.

This dataset is more complex than the number datasets (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.

### Pre-processed Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.

<img src='assets/processed_face_data.png' width=60% />

> If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed_celeba_small/`


```python
# can comment out after executing
# !unzip processed_celeba_small.zip
```


```python
data_dir = 'processed_celeba_small/'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
#import helper

%matplotlib inline
```

## Visualize the CelebA Data

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.

### Pre-process and Load the Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA data.

> There are a few other steps that you'll need to **transform** this data and create a **DataLoader**.

#### Exercise: Complete the following `get_dataloader` function, such that it satisfies these requirements:

* Your images should be square, Tensor images of size `image_size x image_size` in the x and y dimension.
* Your function should return a DataLoader that shuffles and batches these Tensor images.

#### ImageFolder

To create a dataset given a directory of images, it's recommended that you use PyTorch's [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) wrapper, with a root directory `processed_celeba_small/` and data transformation passed in.


```python
# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms
```


```python
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    transform = transforms.Compose([transforms.Resize(image_size),
                                   transforms.ToTensor()]) 
    image_dataset = datasets.ImageFolder(data_dir, transform)
    print (image_dataset)
    return torch.utils.data.DataLoader(image_dataset, batch_size = batch_size, shuffle=True)
```

## Create a DataLoader

#### Exercise: Create a DataLoader `celeba_train_loader` with appropriate hyperparameters.

Call the above function and create a dataloader to view images. 
* You can decide on any reasonable `batch_size` parameter
* Your `image_size` **must be** `32`. Resizing the data to a smaller size will make for faster training, while still creating convincing images of faces!


```python
# Define function hyperparameters
batch_size = 128
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)
```

    Dataset ImageFolder
        Number of datapoints: 89931
        Root Location: processed_celeba_small/
        Transforms (if any): Compose(
                                 Resize(size=32, interpolation=PIL.Image.BILINEAR)
                                 ToTensor()
                             )
        Target Transforms (if any): None


Next, you can view some images! You should seen square images of somewhat-centered faces.

Note: You'll need to convert the Tensor images into a NumPy type and transpose the dimensions to correctly display an image, suggested `imshow` code is below, but it may not be perfect.


```python
# helper display function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
```


![png](output_9_0.png)


#### Exercise: Pre-process your image data and scale it to a pixel range of -1 to 1

You need to do a bit of pre-processing; you know that the output of a `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)


```python
# TODO: Complete the scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x   
#     min_val, max_val = feature_range
#     return x * (max_val - min_val) + min_val
    return x*2-1         #it's always called with -1,1 so for speed

```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# check scaled range
# should be close to -1 to 1
img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())
```

    Min:  tensor(-0.9059)
    Max:  tensor(0.8588)


---
# Define the Model

A GAN is comprised of two adversarial networks, a discriminator and a generator.

## Discriminator

Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with **normalization**. You are also allowed to create any helper functions that may be useful.

#### Exercise: Complete the Discriminator class
* The inputs to the discriminator are 32x32x3 tensor images
* The output should be a single value that will indicate whether a given image is real or fake



```python
import torch.nn as nn
import torch.nn.functional as F
```


```python
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
```


```python
class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # (16, 16, conv_dim)
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (8, 8, conv_dim*2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (4, 4, conv_dim*4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (2, 2, conv_dim*8) 
        
        self.classifier = nn.Linear(conv_dim*8*2*2, 1)

        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        out = F.selu(self.conv1(x), 0.2)
        out = F.selu(self.conv2(out), 0.2)
        out = F.selu(self.conv3(out), 0.2)
        out = F.selu(self.conv4(out), 0.2)
        
        out = out.view(-1, self.conv_dim*8*2*2)
        out = self.classifier(out)
        return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(Discriminator)
```

    Tests Passed


## Generator

The generator should upsample an input and generate a *new* image of the same size as our training data `32x32x3`. This should be mostly transpose convolutional layers with normalization applied to the outputs.

#### Exercise: Complete the Generator class
* The inputs to the generator are vectors of some length `z_size`
* The output should be a image of shape `32x32x3`


```python
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
```


```python
class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim*8*2*2)
        
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4)
        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*8, 2, 2) # (batch_size, depth, 4, 4)
        
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        
        # last layer: tanh activation instead of relu
        out = self.t_conv4(out)
        out = F.tanh(out)
        
        return out

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(Generator)
```

    Tests Passed


## Initialize the weights of your networks

To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), they say:
> All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, your next task will be to define a weight initialization function that does just this!

You can refer back to the lesson on weight initialization or even consult existing model code, such as that from [the `networks.py` file in CycleGAN Github repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py) to help you complete this function.

#### Exercise: Complete the weight initialization function

* This should initialize only **convolutional** and **linear** layers
* Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
* The bias terms, if they exist, may be left alone or set to 0.


```python
from torch.nn import init

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model: convolutional and linear
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    isConvolution = classname.find('Conv') != -1
    isLinear = classname.find('Linear') != -1
    if (hasattr(m, 'weight') and isConvolution or isLinear):
        init.normal_(m.weight.data, 0.0, 0.02)
```

## Build complete network

Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G

```

#### Exercise: Define model hyperparameters


```python
# Define model hyperparams
d_conv_dim = 64
g_conv_dim = 64
z_size = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
```

    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Linear(in_features=2048, out_features=1, bias=True)
    )
    
    Generator(
      (fc): Linear(in_features=100, out_features=2048, bias=True)
      (t_conv1): Sequential(
        (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv2): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv3): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv4): Sequential(
        (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
    )


### Training on GPU

Check if you can train on GPU. Here, we'll set this as a boolean variable `train_on_gpu`. Later, you'll be responsible for making sure that 
>* Models,
* Model inputs, and
* Loss function arguments

Are moved to GPU, where appropriate.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')
```

    Training on GPU!


---
## Discriminator and Generator Losses

Now we need to calculate the losses for both types of adversarial networks.

### Discriminator Losses

> * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`. 
* Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.


### Generator Loss

The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to *think* its generated images are *real*.

#### Exercise: Complete real and fake loss functions

**You may choose to use either cross entropy or a least squares error loss to complete the following `real_loss` and `fake_loss` functions.**


```python
def real_loss(D_out):
    '''
    Calculates how close discriminator outputs are to being real.
    param, D_out: discriminator logits
    return: real loss
    '''
    labels = torch.ones(D_out.size(0))      #D_out[0] is batch size
    if train_on_gpu:
        labels = labels.cuda()
    
    criterion = nn.BCEWithLogitsLoss()
    return criterion(D_out.squeeze(), labels)

def fake_loss(D_out):
    '''
    Calculates how close discriminator outputs are to being fake.
    param, D_out: discriminator logits
    return: fake loss
    '''
    labels = torch.zeros(D_out.size(0))     #D_out[0] is batch size
    if train_on_gpu:
        labels = labels.cuda()
    
    criterion = nn.BCEWithLogitsLoss()
    return criterion(D_out.squeeze(), labels)
```

## Optimizers

#### Exercise: Define optimizers for your Discriminator (D) and Generator (G)

Define optimizers for your models with appropriate hyperparameters.


```python
import torch.optim as optim

lr = 0.0002
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator D and generator G
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
```

---
## Training

Training will involve alternating between training the discriminator and the generator. You'll use your functions `real_loss` and `fake_loss` to help you calculate the discriminator losses.

* You should train the discriminator by alternating on real and fake images
* Then the generator, which tries to trick the discriminator and should have an opposing loss function


#### Saving Samples

You've been given some code to print out some loss statistics and save some generated "fake" samples.

#### Exercise: Complete the training function

Keep in mind that, if you've moved your models to GPU, you'll also have to move any model inputs to GPU.


```python
from workspace_utils import active_session
```


```python
def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================
            
            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()
            if train_on_gpu:
                real_images = real_images.cuda()
                
            D_real = D(real_images)
            d_real_loss = real_loss(D_real)
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()
            
            z = np.random.uniform(-1, 1, size = (batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            
            fake_images = G(z)
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)
            
            g_loss.backward()
            g_optimizer.step()
            
            
            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | batch: {:5d} | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, batch_i+1, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        torch.save(G.state_dict(), 'g'+str(epoch+1)+'.pt')
        torch.save(D.state_dict(), 'd'+str(epoch+1)+'.pt')

        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode
        print('---------------Epoch [{:5d}/{:5d}]---------------'.format(epoch+1, n_epochs))
    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses
```

Set your number of training epochs and train your GAN!


```python
# G.load_state_dict(torch.load('g.pt'))
# D.load_state_dict(torch.load('d.pt'))
```


```python
# set number of epochs 
n_epochs = 20

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# call training function
with active_session():
    losses = train(D, G, n_epochs=n_epochs)
```

    Epoch [    1/   20] | batch:     1 | d_loss: 0.8111 | g_loss: 1.2661
    Epoch [    1/   20] | batch:    51 | d_loss: 0.9952 | g_loss: 2.3659
    Epoch [    1/   20] | batch:   101 | d_loss: 0.9237 | g_loss: 2.7058
    Epoch [    1/   20] | batch:   151 | d_loss: 0.9340 | g_loss: 1.3858
    Epoch [    1/   20] | batch:   201 | d_loss: 0.9927 | g_loss: 1.6645
    Epoch [    1/   20] | batch:   251 | d_loss: 0.8794 | g_loss: 1.3428
    Epoch [    1/   20] | batch:   301 | d_loss: 1.0240 | g_loss: 1.8747
    Epoch [    1/   20] | batch:   351 | d_loss: 0.8578 | g_loss: 2.3406
    Epoch [    1/   20] | batch:   401 | d_loss: 1.0471 | g_loss: 1.4166
    Epoch [    1/   20] | batch:   451 | d_loss: 0.8953 | g_loss: 2.2885
    Epoch [    1/   20] | batch:   501 | d_loss: 0.8606 | g_loss: 1.5577
    Epoch [    1/   20] | batch:   551 | d_loss: 0.8905 | g_loss: 1.9201
    Epoch [    1/   20] | batch:   601 | d_loss: 0.9602 | g_loss: 2.0931
    Epoch [    1/   20] | batch:   651 | d_loss: 0.9449 | g_loss: 2.2206
    Epoch [    1/   20] | batch:   701 | d_loss: 1.0528 | g_loss: 1.5514
    ---------------Epoch [    1/   20]---------------
    Epoch [    2/   20] | batch:     1 | d_loss: 0.8771 | g_loss: 1.3734
    Epoch [    2/   20] | batch:    51 | d_loss: 1.0019 | g_loss: 1.2820
    Epoch [    2/   20] | batch:   101 | d_loss: 0.9900 | g_loss: 1.6812
    Epoch [    2/   20] | batch:   151 | d_loss: 1.0659 | g_loss: 2.1863
    Epoch [    2/   20] | batch:   201 | d_loss: 0.9917 | g_loss: 1.6778
    Epoch [    2/   20] | batch:   251 | d_loss: 0.8652 | g_loss: 1.5668
    Epoch [    2/   20] | batch:   301 | d_loss: 0.7992 | g_loss: 2.3568
    Epoch [    2/   20] | batch:   351 | d_loss: 1.0523 | g_loss: 1.2216
    Epoch [    2/   20] | batch:   401 | d_loss: 1.0096 | g_loss: 2.4254
    Epoch [    2/   20] | batch:   451 | d_loss: 1.1680 | g_loss: 1.1488
    Epoch [    2/   20] | batch:   501 | d_loss: 0.9187 | g_loss: 1.2972
    Epoch [    2/   20] | batch:   551 | d_loss: 0.9080 | g_loss: 1.9679
    Epoch [    2/   20] | batch:   601 | d_loss: 1.0048 | g_loss: 2.4279
    Epoch [    2/   20] | batch:   651 | d_loss: 1.2464 | g_loss: 1.1955
    Epoch [    2/   20] | batch:   701 | d_loss: 1.0181 | g_loss: 2.1282
    ---------------Epoch [    2/   20]---------------
    Epoch [    3/   20] | batch:     1 | d_loss: 0.9845 | g_loss: 3.0135
    Epoch [    3/   20] | batch:    51 | d_loss: 0.8247 | g_loss: 2.0062
    Epoch [    3/   20] | batch:   101 | d_loss: 0.9052 | g_loss: 2.0246
    Epoch [    3/   20] | batch:   151 | d_loss: 0.8806 | g_loss: 1.8005
    Epoch [    3/   20] | batch:   201 | d_loss: 1.1608 | g_loss: 1.1466
    Epoch [    3/   20] | batch:   251 | d_loss: 1.0030 | g_loss: 1.7211
    Epoch [    3/   20] | batch:   301 | d_loss: 1.2835 | g_loss: 1.8969
    Epoch [    3/   20] | batch:   351 | d_loss: 0.8584 | g_loss: 1.6190
    Epoch [    3/   20] | batch:   401 | d_loss: 0.7295 | g_loss: 1.6601
    Epoch [    3/   20] | batch:   451 | d_loss: 1.0188 | g_loss: 1.8397
    Epoch [    3/   20] | batch:   501 | d_loss: 1.2436 | g_loss: 2.5559
    Epoch [    3/   20] | batch:   551 | d_loss: 1.0364 | g_loss: 2.0659
    Epoch [    3/   20] | batch:   601 | d_loss: 1.0645 | g_loss: 1.2833
    Epoch [    3/   20] | batch:   651 | d_loss: 0.8505 | g_loss: 2.2852
    Epoch [    3/   20] | batch:   701 | d_loss: 0.9450 | g_loss: 1.7768
    ---------------Epoch [    3/   20]---------------
    Epoch [    4/   20] | batch:     1 | d_loss: 0.9740 | g_loss: 1.1168
    Epoch [    4/   20] | batch:    51 | d_loss: 0.9093 | g_loss: 1.2597
    Epoch [    4/   20] | batch:   101 | d_loss: 1.0745 | g_loss: 0.9085
    Epoch [    4/   20] | batch:   151 | d_loss: 0.7689 | g_loss: 1.8919
    Epoch [    4/   20] | batch:   201 | d_loss: 1.2617 | g_loss: 3.2608
    Epoch [    4/   20] | batch:   251 | d_loss: 1.0825 | g_loss: 1.3975
    Epoch [    4/   20] | batch:   301 | d_loss: 1.0497 | g_loss: 0.9826
    Epoch [    4/   20] | batch:   351 | d_loss: 0.8368 | g_loss: 1.8344
    Epoch [    4/   20] | batch:   401 | d_loss: 0.8927 | g_loss: 1.4510
    Epoch [    4/   20] | batch:   451 | d_loss: 0.8737 | g_loss: 2.1484
    Epoch [    4/   20] | batch:   501 | d_loss: 0.8620 | g_loss: 1.7640
    Epoch [    4/   20] | batch:   551 | d_loss: 1.0488 | g_loss: 1.1829
    Epoch [    4/   20] | batch:   601 | d_loss: 0.9488 | g_loss: 1.2433
    Epoch [    4/   20] | batch:   651 | d_loss: 0.8552 | g_loss: 2.1256
    Epoch [    4/   20] | batch:   701 | d_loss: 1.2086 | g_loss: 1.2018
    ---------------Epoch [    4/   20]---------------
    Epoch [    5/   20] | batch:     1 | d_loss: 0.6779 | g_loss: 2.0733
    Epoch [    5/   20] | batch:    51 | d_loss: 0.9240 | g_loss: 0.9773
    Epoch [    5/   20] | batch:   101 | d_loss: 0.8876 | g_loss: 1.7137
    Epoch [    5/   20] | batch:   151 | d_loss: 1.0030 | g_loss: 2.4516
    Epoch [    5/   20] | batch:   201 | d_loss: 0.9148 | g_loss: 1.8731
    Epoch [    5/   20] | batch:   251 | d_loss: 0.9607 | g_loss: 1.9129
    Epoch [    5/   20] | batch:   301 | d_loss: 0.9252 | g_loss: 1.8296
    Epoch [    5/   20] | batch:   351 | d_loss: 0.7578 | g_loss: 2.0952
    Epoch [    5/   20] | batch:   401 | d_loss: 0.7759 | g_loss: 1.5163
    Epoch [    5/   20] | batch:   451 | d_loss: 0.8367 | g_loss: 1.9044
    Epoch [    5/   20] | batch:   501 | d_loss: 0.8096 | g_loss: 1.8013
    Epoch [    5/   20] | batch:   551 | d_loss: 0.7723 | g_loss: 1.8913
    Epoch [    5/   20] | batch:   601 | d_loss: 0.7605 | g_loss: 2.4772
    Epoch [    5/   20] | batch:   651 | d_loss: 1.3507 | g_loss: 1.2231
    Epoch [    5/   20] | batch:   701 | d_loss: 0.7732 | g_loss: 2.0357
    ---------------Epoch [    5/   20]---------------
    Epoch [    6/   20] | batch:     1 | d_loss: 0.8181 | g_loss: 1.4688
    Epoch [    6/   20] | batch:    51 | d_loss: 1.3336 | g_loss: 1.3665
    Epoch [    6/   20] | batch:   101 | d_loss: 0.8859 | g_loss: 1.9091
    Epoch [    6/   20] | batch:   151 | d_loss: 0.8380 | g_loss: 2.1336
    Epoch [    6/   20] | batch:   201 | d_loss: 0.8269 | g_loss: 3.3221
    Epoch [    6/   20] | batch:   251 | d_loss: 0.9188 | g_loss: 1.9334
    Epoch [    6/   20] | batch:   301 | d_loss: 0.9195 | g_loss: 1.7430
    Epoch [    6/   20] | batch:   351 | d_loss: 1.2432 | g_loss: 3.4875
    Epoch [    6/   20] | batch:   401 | d_loss: 0.7508 | g_loss: 1.7305
    Epoch [    6/   20] | batch:   451 | d_loss: 0.9282 | g_loss: 1.9829
    Epoch [    6/   20] | batch:   501 | d_loss: 0.8910 | g_loss: 2.0563
    Epoch [    6/   20] | batch:   551 | d_loss: 0.7907 | g_loss: 1.6141
    Epoch [    6/   20] | batch:   601 | d_loss: 0.8585 | g_loss: 2.6222
    Epoch [    6/   20] | batch:   651 | d_loss: 0.7723 | g_loss: 1.0951
    Epoch [    6/   20] | batch:   701 | d_loss: 0.8480 | g_loss: 2.0792
    ---------------Epoch [    6/   20]---------------
    Epoch [    7/   20] | batch:     1 | d_loss: 0.8413 | g_loss: 1.3066
    Epoch [    7/   20] | batch:    51 | d_loss: 0.7940 | g_loss: 2.6432
    Epoch [    7/   20] | batch:   101 | d_loss: 0.8089 | g_loss: 1.8077
    Epoch [    7/   20] | batch:   151 | d_loss: 0.9595 | g_loss: 1.6319
    Epoch [    7/   20] | batch:   201 | d_loss: 1.0729 | g_loss: 0.9551
    Epoch [    7/   20] | batch:   251 | d_loss: 0.9064 | g_loss: 2.4957
    Epoch [    7/   20] | batch:   301 | d_loss: 0.7105 | g_loss: 2.1476
    Epoch [    7/   20] | batch:   351 | d_loss: 1.0778 | g_loss: 2.9749
    Epoch [    7/   20] | batch:   401 | d_loss: 0.9246 | g_loss: 1.3092
    Epoch [    7/   20] | batch:   451 | d_loss: 1.0882 | g_loss: 0.7023
    Epoch [    7/   20] | batch:   501 | d_loss: 1.0619 | g_loss: 1.6870
    Epoch [    7/   20] | batch:   551 | d_loss: 0.8053 | g_loss: 2.0815
    Epoch [    7/   20] | batch:   601 | d_loss: 0.7979 | g_loss: 2.2491
    Epoch [    7/   20] | batch:   651 | d_loss: 0.8395 | g_loss: 1.9564
    Epoch [    7/   20] | batch:   701 | d_loss: 1.0466 | g_loss: 1.3503
    ---------------Epoch [    7/   20]---------------
    Epoch [    8/   20] | batch:     1 | d_loss: 0.6671 | g_loss: 1.4386
    Epoch [    8/   20] | batch:    51 | d_loss: 0.9737 | g_loss: 1.9353
    Epoch [    8/   20] | batch:   101 | d_loss: 0.7338 | g_loss: 1.6522
    Epoch [    8/   20] | batch:   151 | d_loss: 0.8512 | g_loss: 2.1538
    Epoch [    8/   20] | batch:   201 | d_loss: 0.7490 | g_loss: 2.2567
    Epoch [    8/   20] | batch:   251 | d_loss: 0.8791 | g_loss: 2.4892
    Epoch [    8/   20] | batch:   301 | d_loss: 0.7260 | g_loss: 1.8245
    Epoch [    8/   20] | batch:   351 | d_loss: 0.8386 | g_loss: 2.1277
    Epoch [    8/   20] | batch:   401 | d_loss: 0.9722 | g_loss: 2.4713
    Epoch [    8/   20] | batch:   451 | d_loss: 0.6980 | g_loss: 2.2593
    Epoch [    8/   20] | batch:   501 | d_loss: 1.0766 | g_loss: 3.4284
    Epoch [    8/   20] | batch:   551 | d_loss: 0.9441 | g_loss: 2.4267
    Epoch [    8/   20] | batch:   601 | d_loss: 0.9615 | g_loss: 1.1553
    Epoch [    8/   20] | batch:   651 | d_loss: 0.9098 | g_loss: 2.4200
    Epoch [    8/   20] | batch:   701 | d_loss: 0.8843 | g_loss: 1.3202
    ---------------Epoch [    8/   20]---------------
    Epoch [    9/   20] | batch:     1 | d_loss: 0.7078 | g_loss: 1.8200
    Epoch [    9/   20] | batch:    51 | d_loss: 1.0578 | g_loss: 1.2410
    Epoch [    9/   20] | batch:   101 | d_loss: 0.8762 | g_loss: 1.8029
    Epoch [    9/   20] | batch:   151 | d_loss: 0.8846 | g_loss: 1.5970
    Epoch [    9/   20] | batch:   201 | d_loss: 0.9776 | g_loss: 3.1247
    Epoch [    9/   20] | batch:   251 | d_loss: 0.7748 | g_loss: 1.2802
    Epoch [    9/   20] | batch:   301 | d_loss: 0.7308 | g_loss: 2.0371
    Epoch [    9/   20] | batch:   351 | d_loss: 0.9546 | g_loss: 1.3827
    Epoch [    9/   20] | batch:   401 | d_loss: 0.9979 | g_loss: 2.7320
    Epoch [    9/   20] | batch:   451 | d_loss: 0.8290 | g_loss: 1.6430
    Epoch [    9/   20] | batch:   501 | d_loss: 0.8173 | g_loss: 1.2279
    Epoch [    9/   20] | batch:   551 | d_loss: 0.7015 | g_loss: 1.6330
    Epoch [    9/   20] | batch:   601 | d_loss: 1.2070 | g_loss: 2.3610
    Epoch [    9/   20] | batch:   651 | d_loss: 1.2740 | g_loss: 0.5588
    Epoch [    9/   20] | batch:   701 | d_loss: 0.8654 | g_loss: 3.2853
    ---------------Epoch [    9/   20]---------------
    Epoch [   10/   20] | batch:     1 | d_loss: 1.0506 | g_loss: 1.6972
    Epoch [   10/   20] | batch:    51 | d_loss: 0.8623 | g_loss: 1.2557
    Epoch [   10/   20] | batch:   101 | d_loss: 0.7753 | g_loss: 1.7687
    Epoch [   10/   20] | batch:   151 | d_loss: 0.6656 | g_loss: 1.5877
    Epoch [   10/   20] | batch:   201 | d_loss: 0.9033 | g_loss: 1.9987
    Epoch [   10/   20] | batch:   251 | d_loss: 0.8301 | g_loss: 1.5547
    Epoch [   10/   20] | batch:   301 | d_loss: 0.9235 | g_loss: 2.9890
    Epoch [   10/   20] | batch:   351 | d_loss: 0.8235 | g_loss: 1.8149
    Epoch [   10/   20] | batch:   401 | d_loss: 0.6793 | g_loss: 2.3275
    Epoch [   10/   20] | batch:   451 | d_loss: 0.7816 | g_loss: 3.2800
    Epoch [   10/   20] | batch:   501 | d_loss: 0.7944 | g_loss: 1.7540
    Epoch [   10/   20] | batch:   551 | d_loss: 0.9119 | g_loss: 1.2870
    Epoch [   10/   20] | batch:   601 | d_loss: 1.1382 | g_loss: 2.2872
    Epoch [   10/   20] | batch:   651 | d_loss: 0.8099 | g_loss: 1.2067
    Epoch [   10/   20] | batch:   701 | d_loss: 0.6867 | g_loss: 1.9326
    ---------------Epoch [   10/   20]---------------
    Epoch [   11/   20] | batch:     1 | d_loss: 0.8320 | g_loss: 1.1754
    Epoch [   11/   20] | batch:    51 | d_loss: 0.6943 | g_loss: 1.6403
    Epoch [   11/   20] | batch:   101 | d_loss: 0.9558 | g_loss: 3.0082
    Epoch [   11/   20] | batch:   151 | d_loss: 0.8052 | g_loss: 2.7568
    Epoch [   11/   20] | batch:   201 | d_loss: 0.7848 | g_loss: 1.9918
    Epoch [   11/   20] | batch:   251 | d_loss: 0.8150 | g_loss: 1.4829
    Epoch [   11/   20] | batch:   301 | d_loss: 0.7332 | g_loss: 2.1810
    Epoch [   11/   20] | batch:   351 | d_loss: 0.8433 | g_loss: 1.8746
    Epoch [   11/   20] | batch:   401 | d_loss: 0.6758 | g_loss: 1.3094
    Epoch [   11/   20] | batch:   451 | d_loss: 0.8459 | g_loss: 1.6590
    Epoch [   11/   20] | batch:   501 | d_loss: 0.8478 | g_loss: 1.3339
    Epoch [   11/   20] | batch:   551 | d_loss: 0.8105 | g_loss: 2.5290
    Epoch [   11/   20] | batch:   601 | d_loss: 0.8847 | g_loss: 2.9163
    Epoch [   11/   20] | batch:   651 | d_loss: 0.9158 | g_loss: 3.0656
    Epoch [   11/   20] | batch:   701 | d_loss: 0.7829 | g_loss: 2.7488
    ---------------Epoch [   11/   20]---------------
    Epoch [   12/   20] | batch:     1 | d_loss: 0.7352 | g_loss: 1.9948
    Epoch [   12/   20] | batch:    51 | d_loss: 0.9262 | g_loss: 1.6865
    Epoch [   12/   20] | batch:   101 | d_loss: 0.8878 | g_loss: 2.6026
    Epoch [   12/   20] | batch:   151 | d_loss: 0.7256 | g_loss: 1.7047
    Epoch [   12/   20] | batch:   201 | d_loss: 0.8756 | g_loss: 1.5276
    Epoch [   12/   20] | batch:   251 | d_loss: 1.1679 | g_loss: 0.8444
    Epoch [   12/   20] | batch:   301 | d_loss: 0.6936 | g_loss: 2.6643
    Epoch [   12/   20] | batch:   351 | d_loss: 0.8501 | g_loss: 2.4405
    Epoch [   12/   20] | batch:   401 | d_loss: 0.7047 | g_loss: 2.2575
    Epoch [   12/   20] | batch:   451 | d_loss: 0.6888 | g_loss: 2.6093
    Epoch [   12/   20] | batch:   501 | d_loss: 0.7325 | g_loss: 1.3854
    Epoch [   12/   20] | batch:   551 | d_loss: 1.0565 | g_loss: 3.9395
    Epoch [   12/   20] | batch:   601 | d_loss: 0.7159 | g_loss: 3.6967
    Epoch [   12/   20] | batch:   651 | d_loss: 0.7333 | g_loss: 1.4379
    Epoch [   12/   20] | batch:   701 | d_loss: 0.7963 | g_loss: 1.4610
    ---------------Epoch [   12/   20]---------------
    Epoch [   13/   20] | batch:     1 | d_loss: 0.6865 | g_loss: 2.8950
    Epoch [   13/   20] | batch:    51 | d_loss: 0.9207 | g_loss: 1.5573
    Epoch [   13/   20] | batch:   101 | d_loss: 1.2883 | g_loss: 1.4500
    Epoch [   13/   20] | batch:   151 | d_loss: 0.8684 | g_loss: 2.6025
    Epoch [   13/   20] | batch:   201 | d_loss: 0.7481 | g_loss: 1.6527
    Epoch [   13/   20] | batch:   251 | d_loss: 0.8783 | g_loss: 1.3452
    Epoch [   13/   20] | batch:   301 | d_loss: 0.7375 | g_loss: 2.0968
    Epoch [   13/   20] | batch:   351 | d_loss: 0.6422 | g_loss: 2.5757
    Epoch [   13/   20] | batch:   401 | d_loss: 0.7280 | g_loss: 1.3058
    Epoch [   13/   20] | batch:   451 | d_loss: 0.7422 | g_loss: 1.4346
    Epoch [   13/   20] | batch:   501 | d_loss: 0.9727 | g_loss: 3.5498
    Epoch [   13/   20] | batch:   551 | d_loss: 0.6053 | g_loss: 2.1011
    Epoch [   13/   20] | batch:   601 | d_loss: 1.5844 | g_loss: 0.5962
    Epoch [   13/   20] | batch:   651 | d_loss: 0.5554 | g_loss: 2.5573
    Epoch [   13/   20] | batch:   701 | d_loss: 0.6035 | g_loss: 2.5171
    ---------------Epoch [   13/   20]---------------
    Epoch [   14/   20] | batch:     1 | d_loss: 0.7339 | g_loss: 3.6615
    Epoch [   14/   20] | batch:    51 | d_loss: 0.8485 | g_loss: 2.3583
    Epoch [   14/   20] | batch:   101 | d_loss: 0.6495 | g_loss: 1.8807
    Epoch [   14/   20] | batch:   151 | d_loss: 0.5651 | g_loss: 1.4821
    Epoch [   14/   20] | batch:   201 | d_loss: 0.7518 | g_loss: 2.2164
    Epoch [   14/   20] | batch:   251 | d_loss: 0.7359 | g_loss: 1.7076
    Epoch [   14/   20] | batch:   301 | d_loss: 1.4818 | g_loss: 3.6353
    Epoch [   14/   20] | batch:   351 | d_loss: 0.8072 | g_loss: 3.1063
    Epoch [   14/   20] | batch:   401 | d_loss: 0.6912 | g_loss: 1.5440
    Epoch [   14/   20] | batch:   451 | d_loss: 0.7894 | g_loss: 1.2850
    Epoch [   14/   20] | batch:   501 | d_loss: 0.8157 | g_loss: 3.7143
    Epoch [   14/   20] | batch:   551 | d_loss: 0.8888 | g_loss: 4.6283
    Epoch [   14/   20] | batch:   601 | d_loss: 0.7662 | g_loss: 1.5938
    Epoch [   14/   20] | batch:   651 | d_loss: 0.8177 | g_loss: 1.5437
    Epoch [   14/   20] | batch:   701 | d_loss: 1.0911 | g_loss: 1.2381
    ---------------Epoch [   14/   20]---------------
    Epoch [   15/   20] | batch:     1 | d_loss: 0.6332 | g_loss: 2.4432
    Epoch [   15/   20] | batch:    51 | d_loss: 0.5178 | g_loss: 2.3152
    Epoch [   15/   20] | batch:   101 | d_loss: 0.5655 | g_loss: 2.4855
    Epoch [   15/   20] | batch:   151 | d_loss: 1.4360 | g_loss: 4.9096
    Epoch [   15/   20] | batch:   201 | d_loss: 0.6065 | g_loss: 1.9198
    Epoch [   15/   20] | batch:   251 | d_loss: 0.6617 | g_loss: 2.7955
    Epoch [   15/   20] | batch:   301 | d_loss: 0.5642 | g_loss: 2.5870
    Epoch [   15/   20] | batch:   351 | d_loss: 0.7548 | g_loss: 1.6740
    Epoch [   15/   20] | batch:   401 | d_loss: 0.7114 | g_loss: 1.2314
    Epoch [   15/   20] | batch:   451 | d_loss: 0.7264 | g_loss: 2.8277
    Epoch [   15/   20] | batch:   501 | d_loss: 0.8249 | g_loss: 1.9137
    Epoch [   15/   20] | batch:   551 | d_loss: 0.6711 | g_loss: 1.9008
    Epoch [   15/   20] | batch:   601 | d_loss: 0.7163 | g_loss: 1.7807
    Epoch [   15/   20] | batch:   651 | d_loss: 0.7018 | g_loss: 1.8669
    Epoch [   15/   20] | batch:   701 | d_loss: 1.2824 | g_loss: 0.8433
    ---------------Epoch [   15/   20]---------------
    Epoch [   16/   20] | batch:     1 | d_loss: 0.7795 | g_loss: 1.9912
    Epoch [   16/   20] | batch:    51 | d_loss: 0.7561 | g_loss: 2.7653
    Epoch [   16/   20] | batch:   101 | d_loss: 0.9359 | g_loss: 1.5608
    Epoch [   16/   20] | batch:   151 | d_loss: 0.5931 | g_loss: 2.3988
    Epoch [   16/   20] | batch:   201 | d_loss: 0.8050 | g_loss: 1.1277
    Epoch [   16/   20] | batch:   251 | d_loss: 0.4762 | g_loss: 2.5125
    Epoch [   16/   20] | batch:   301 | d_loss: 0.6604 | g_loss: 2.3707
    Epoch [   16/   20] | batch:   351 | d_loss: 1.1260 | g_loss: 1.0176
    Epoch [   16/   20] | batch:   401 | d_loss: 0.5498 | g_loss: 2.5035
    Epoch [   16/   20] | batch:   451 | d_loss: 0.6702 | g_loss: 2.3546
    Epoch [   16/   20] | batch:   501 | d_loss: 0.5987 | g_loss: 2.7054
    Epoch [   16/   20] | batch:   551 | d_loss: 0.6563 | g_loss: 1.9455
    Epoch [   16/   20] | batch:   601 | d_loss: 0.4369 | g_loss: 2.3982
    Epoch [   16/   20] | batch:   651 | d_loss: 0.6192 | g_loss: 1.6578
    Epoch [   16/   20] | batch:   701 | d_loss: 0.8847 | g_loss: 3.6035
    ---------------Epoch [   16/   20]---------------
    Epoch [   17/   20] | batch:     1 | d_loss: 0.8319 | g_loss: 1.7466
    Epoch [   17/   20] | batch:    51 | d_loss: 0.8202 | g_loss: 2.0495
    Epoch [   17/   20] | batch:   101 | d_loss: 0.9077 | g_loss: 2.2762
    Epoch [   17/   20] | batch:   151 | d_loss: 0.9110 | g_loss: 1.5152
    Epoch [   17/   20] | batch:   201 | d_loss: 0.6959 | g_loss: 3.1359
    Epoch [   17/   20] | batch:   251 | d_loss: 0.6086 | g_loss: 3.1093
    Epoch [   17/   20] | batch:   301 | d_loss: 0.9669 | g_loss: 3.7573
    Epoch [   17/   20] | batch:   351 | d_loss: 0.6061 | g_loss: 2.2916
    Epoch [   17/   20] | batch:   401 | d_loss: 1.3833 | g_loss: 0.8206
    Epoch [   17/   20] | batch:   451 | d_loss: 0.7337 | g_loss: 1.2652
    Epoch [   17/   20] | batch:   501 | d_loss: 0.5545 | g_loss: 2.2966
    Epoch [   17/   20] | batch:   551 | d_loss: 0.4312 | g_loss: 2.7339
    Epoch [   17/   20] | batch:   601 | d_loss: 1.2815 | g_loss: 4.5698
    Epoch [   17/   20] | batch:   651 | d_loss: 0.5700 | g_loss: 2.5264
    Epoch [   17/   20] | batch:   701 | d_loss: 0.6704 | g_loss: 3.2834
    ---------------Epoch [   17/   20]---------------
    Epoch [   18/   20] | batch:     1 | d_loss: 0.4970 | g_loss: 2.2410
    Epoch [   18/   20] | batch:    51 | d_loss: 0.6295 | g_loss: 4.3854
    Epoch [   18/   20] | batch:   101 | d_loss: 0.6976 | g_loss: 4.3022
    Epoch [   18/   20] | batch:   151 | d_loss: 1.0637 | g_loss: 2.3074
    Epoch [   18/   20] | batch:   201 | d_loss: 0.6216 | g_loss: 1.5955
    Epoch [   18/   20] | batch:   251 | d_loss: 0.6220 | g_loss: 2.0259
    Epoch [   18/   20] | batch:   301 | d_loss: 0.4838 | g_loss: 2.7805
    Epoch [   18/   20] | batch:   351 | d_loss: 0.7369 | g_loss: 3.7731
    Epoch [   18/   20] | batch:   401 | d_loss: 0.6132 | g_loss: 1.4443
    Epoch [   18/   20] | batch:   451 | d_loss: 0.5376 | g_loss: 3.1501
    Epoch [   18/   20] | batch:   501 | d_loss: 0.4992 | g_loss: 1.8216
    Epoch [   18/   20] | batch:   551 | d_loss: 0.9730 | g_loss: 3.3499
    Epoch [   18/   20] | batch:   601 | d_loss: 0.4792 | g_loss: 2.8586
    Epoch [   18/   20] | batch:   651 | d_loss: 0.6218 | g_loss: 2.6634
    Epoch [   18/   20] | batch:   701 | d_loss: 0.6697 | g_loss: 2.0219
    ---------------Epoch [   18/   20]---------------
    Epoch [   19/   20] | batch:     1 | d_loss: 0.5197 | g_loss: 1.8495
    Epoch [   19/   20] | batch:    51 | d_loss: 0.5086 | g_loss: 1.6420
    Epoch [   19/   20] | batch:   101 | d_loss: 0.6986 | g_loss: 1.1413
    Epoch [   19/   20] | batch:   151 | d_loss: 0.6020 | g_loss: 1.8165
    Epoch [   19/   20] | batch:   201 | d_loss: 0.9451 | g_loss: 1.4040
    Epoch [   19/   20] | batch:   251 | d_loss: 0.4777 | g_loss: 3.1983
    Epoch [   19/   20] | batch:   301 | d_loss: 0.9183 | g_loss: 4.0350
    Epoch [   19/   20] | batch:   351 | d_loss: 0.4961 | g_loss: 2.3933
    Epoch [   19/   20] | batch:   401 | d_loss: 0.8045 | g_loss: 1.3214
    Epoch [   19/   20] | batch:   451 | d_loss: 0.5569 | g_loss: 2.1318
    Epoch [   19/   20] | batch:   501 | d_loss: 0.5449 | g_loss: 2.9328
    Epoch [   19/   20] | batch:   551 | d_loss: 0.5860 | g_loss: 1.8094
    Epoch [   19/   20] | batch:   601 | d_loss: 0.5079 | g_loss: 1.9517
    Epoch [   19/   20] | batch:   651 | d_loss: 0.5289 | g_loss: 2.2257
    Epoch [   19/   20] | batch:   701 | d_loss: 0.5177 | g_loss: 1.4699
    ---------------Epoch [   19/   20]---------------
    Epoch [   20/   20] | batch:     1 | d_loss: 0.8772 | g_loss: 3.8187
    Epoch [   20/   20] | batch:    51 | d_loss: 1.0291 | g_loss: 4.1244
    Epoch [   20/   20] | batch:   101 | d_loss: 0.4135 | g_loss: 2.1966
    Epoch [   20/   20] | batch:   151 | d_loss: 0.4319 | g_loss: 2.7543
    Epoch [   20/   20] | batch:   201 | d_loss: 0.5398 | g_loss: 2.8416
    Epoch [   20/   20] | batch:   251 | d_loss: 0.4549 | g_loss: 2.5017
    Epoch [   20/   20] | batch:   301 | d_loss: 0.6065 | g_loss: 2.1791
    Epoch [   20/   20] | batch:   351 | d_loss: 0.3795 | g_loss: 3.5890
    Epoch [   20/   20] | batch:   401 | d_loss: 0.6528 | g_loss: 1.6973
    Epoch [   20/   20] | batch:   451 | d_loss: 0.5359 | g_loss: 1.7315
    Epoch [   20/   20] | batch:   501 | d_loss: 0.3617 | g_loss: 3.3402
    Epoch [   20/   20] | batch:   551 | d_loss: 0.5715 | g_loss: 2.3122
    Epoch [   20/   20] | batch:   601 | d_loss: 0.5038 | g_loss: 2.7294
    Epoch [   20/   20] | batch:   651 | d_loss: 0.4265 | g_loss: 2.0100
    Epoch [   20/   20] | batch:   701 | d_loss: 0.5678 | g_loss: 1.9734
    ---------------Epoch [   20/   20]---------------


## Training loss

Plot the training losses for the generator and discriminator, recorded after each epoch.


```python
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f13f4acc128>




![png](output_40_1.png)


## Generator samples from training

View samples of images from the generator, and answer a question about the strengths and weaknesses of your trained models.


```python
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
```


```python
# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
```


```python
view_samples(-1, samples)
```


![png](output_44_0.png)


### Question: What do you notice about your generated samples and how might you improve this model?
When you answer this question, consider the following factors:
* The dataset is biased; it is made of "celebrity" faces that are mostly white
* Model size; larger models have the opportunity to learn more features in a data feature space
* Optimization strategy; optimizers and number of epochs affect your final result


**Answer:** 
The training data does not have the complete face. Features like chins are not visible. As a results, the generated images miss chins. Data should preferably have the complete face.

Obtaining higher resolution images. I would like to study NVIDIA's face generation paper where they trained HD faces and can even merge two faces to crate a new one.

A larger model size with more hidden dimensions would work better but take more training time

Training for longer will be ueful as the discriminator loss is steadily decreasing anf the generator loss curve is jagged indicating, it is imagining new features well, many of which are incorrect, but that's a step in the right direction.

### Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "problem_unittests.py" files in your submission.
