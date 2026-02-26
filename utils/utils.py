import torch
import wandb
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from netCDF4 import Dataset as d
import matplotlib.pyplot as plt
import numpy as np

def log_wandb(c, lam):
    wandb.login()

    run = wandb.init(entity="vadim-limousin-new-york-university",
                     project="Strat-PINN",
                     name=c.name+" %s %s" % (c.nb_buoys, lam),
                     config={
                         "wt": c.wt,
                         "lam": lam,
                         "kx": c.kx,
                         "nb_epochs": c.nb_epochs,
                         "nb_rounds": c.nb_rounds,
                         "nb_buoys": c.nb_buoys,
                     },
                     )

    return run

def read_data():
    global variable_data

    # Read the vars.nc file
    nc_file = './Data/Dataset_oceano/vars.nc'
    nc_dataset = d(nc_file, 'r')

    # Get access to the variables of interest
    variable_data = nc_dataset.variables['psi'][:]

    nc_dataset.close()# Read the vars.nc file
    nc_file = './Data/Dataset_oceano/vars.nc'
    nc_dataset = d(nc_file, 'r')

    # Get access to the variables of interest
    variable_data = nc_dataset.variables['psi'][:]

    nc_dataset.close()

def get_flow_tensor(sidelen, timelen, level, mean, std):
    img = Image.fromarray(variable_data[0:timelen][:, level].reshape(513*timelen, 513))
    transform = Compose([
        Resize((timelen*sidelen, sidelen)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    img = transform(img)
    img = img.reshape(timelen, sidelen, sidelen)
    return img

def get_val_flow_tensor(sidelen, timelen, level, mean, std):
    jump = 100//timelen
    img = Image.fromarray(variable_data[0:100:jump][:, level].reshape(513*timelen, 513))
    transform = Compose([
        Resize((timelen*sidelen, sidelen)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    img = transform(img)
    img = img.reshape(timelen, sidelen, sidelen)
    return img

def get_grid(sidelen, timelen):
    """
        Définit la grille (x, y, t)
    """
    x = torch.linspace(-1, 1, steps=sidelen)
    y = torch.linspace(-1, 1, steps=sidelen)
    t = torch.linspace(-1, 1, steps=timelen)
    mgrid = torch.stack(torch.meshgrid(t, x, y, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, 3)
    mgrid = torch.stack((mgrid[:, 0], mgrid[:, 2], mgrid[:, 1]), dim=-1)
    return mgrid

def get_data():
    data_psi1 = torch.load('./Data/Dataset_oceano_masked/513_short/psi1')

    mask = torch.load('./Data/Dataset_oceano_masked/513_short/mask')
    coords = torch.load('./Data/Dataset_oceano_masked/513_short/coords')
    std = torch.load('./Data/Dataset_oceano_masked/513_short/std')
    return data_psi1, coords, mask, std

def quick_val(params, psi, level, res, plot=False):
    gt = get_val_flow_tensor(*res, level, params.mean, params.std).to(params.device)
    grid = get_grid(*res).to(params.device)

    with torch.no_grad():
        output, _ = psi(grid)
        output = output.reshape(res[1], res[0], res[0])
        mse = (gt - output).pow(2).mean()

    if plot:
        V = np.quantile(gt.to('cpu'), 0.99)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        I = ax1.imshow(gt.to('cpu').reshape(res[1], res[0], res[0])[0], cmap="seismic", origin='lower', vmin=-V, vmax=V)
        ax2.imshow(output.to('cpu').reshape(res[1], res[0], res[0])[0], cmap="seismic", origin='lower', vmin=-V, vmax=V)
        plt.colorbar(I, ax=(ax1, ax2))
        params.run.log({"Output psi %s"%(level+1): wandb.Image(fig)})

    return mse