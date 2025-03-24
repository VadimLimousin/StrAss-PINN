from SineLayer import Siren
from Dynamics import *

from netCDF4 import Dataset as d

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np

from tqdm import tqdm

###
    # Load dataset
###

# Read the vars.nc file
nc_file = './Data/Dataset_oceano/vars.nc'
nc_dataset = d(nc_file, 'r')

# Get access to the variables of interest
variable_data = nc_dataset.variables['psi'][:]

nc_dataset.close()

###
    # The model
###

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
    data_psi1 = torch.load('./Data/Dataset_oceano_masked/psi1')

    mask = torch.load('./Data/Dataset_oceano_masked/mask')
    coords = torch.load('./Data/Dataset_oceano_masked/coords')
    std = torch.load('./Data/Dataset_oceano_masked/std')
    return data_psi1, coords, mask, std

img_size = 513
t_size = 100
nb = 800 # number of argo buoys in the domain

# Learning hyperparam:

batch_size_reg = 20480
nb_reg = 13*batch_size_reg

nb_surface = 392470
batch_size_surf = nb_surface//13

nb_int = t_size*800
batch_size_int = nb_int//10

nb_bot = t_size*800//10
batch_size_bot = nb_bot//10

class SurfaceFitting(Dataset):
    def __init__(self, sidelen, timelen, psi, coords, mask, noise=None):
        super().__init__()
        self.shape = (timelen, sidelen, sidelen)
        self.coords = coords
        self.mask = mask
        self.noise = noise_strength*torch.randn(psi.shape) if noise is None else noise
        self.psi = psi if self.noise is None else psi + self.noise

    def __len__(self):
        return len(self.psi)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError
        return self.coords[idx], self.psi[idx]

class ImageRandomMask(Dataset):
    def __init__(self, sidelen, timelen, level, nb_points, idx=None, noise=None):
        if idx is None:
            self.idx = torch.zeros(nb_points, dtype=torch.int)
            t_max = nb_points//nb
            for t in range(0, t_max):
                self.idx[t*nb:(t+1)*nb] = sidelen*sidelen*t*timelen//t_max + torch.randperm(sidelen*sidelen)[:nb]
        else:
            self.idx = idx
        super().__init__()
        psi = get_flow_tensor(sidelen, timelen, level, mean, std)
        psi = psi.view(-1, 1)[self.idx]
        self.noise = noise_strength*torch.randn(psi.shape) if noise is None else noise
        self.psi = psi if self.noise is None else psi + self.noise
        self.coords = get_grid(sidelen, timelen)[self.idx, :]

    def __len__(self):
        return len(self.psi)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError

        return self.coords[idx], self.psi[idx]

class Regularizer(Dataset):
    def __init__(self, nb_reg, level):
        super().__init__()
        self.coords = torch.zeros(nb_reg, 3).uniform_(-1, 1) # TESTER AVEC LA GRILLE
        self.gradpsi_below = torch.zeros(nb_reg, 3).view(-1, 3)
        self.gradpsi_above = torch.zeros(nb_reg, 3).view(-1, 3)
        self.level = level
        self.lappsi_above = torch.zeros(nb_reg)
        self.lappsi_below = torch.zeros(nb_reg)
        if self.level==0:
            self.forc = wind_forc(self.coords)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError

        if self.level==0:
            return self.coords[idx], self.gradpsi_below[idx], self.lappsi_below[idx], self.forc[idx]
        elif self.level==1:
            return self.coords[idx], self.gradpsi_above[idx], self.lappsi_above[idx], self.gradpsi_below[idx], self.lappsi_below[idx]
        else:
            return self.coords[idx], self.gradpsi_above[idx], self.lappsi_above[idx]

# Before submitting job verify you have chosen the following parameters and maybe written them in the file name "end".

hidden_layers = 2
nb_neurons = 125
wt = 1.5
kx = 10
LAM = [0, 1e-4, 1]
nb_epochs = 400
nb_rounds = 3 # number of iterations of the block strategy
noise_strength = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("")
print("RECAP:")
print("")
print("SIREN model")
print("nb layers:", 2+hidden_layers)
print("nb neurons per layer:", nb_neurons)
print("wt:", wt)
print("kx:", kx)
print("")
print("Learning")
print("Running on:", device)
print("Number of trainable parameters:", 4*nb_neurons + (nb_neurons+1)*nb_neurons*hidden_layers + nb_neurons+1)
print("batch_size_surf:", batch_size_surf)
print("batch_size_int:", batch_size_int)
print("batch_size_bot:", batch_size_bot)
print("lam:", LAM)
print("nb_epochs:", nb_epochs)
print("nb_rounds:", nb_rounds)
print("noise strength in the data:", noise_strength)
print("")
print("Save name:", "_StrAss-PINN_lam=LAM_"+str(nb_neurons))

for lam in LAM:
    end = "_StrAss-PINN_lam=%s"%lam+"_"+str(nb_neurons)

    psi1, coords1, mask, std = get_data()
    mean = psi1.mean()

    data_psi1 = SurfaceFitting(img_size, t_size, psi1, coords1, mask) # put noise=0 if you want no noise
    regdata_psi1 = Regularizer(nb_reg, 0)
    dataloader1 = DataLoader(data_psi1, batch_size=batch_size_surf, pin_memory=False, num_workers=0, shuffle=True) # observation points
    regdataloader1 = DataLoader(regdata_psi1, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True) # collocation points

    noise1 = data_psi1.noise

    psi1 = Siren(in_features=3, out_features=1, hidden_features=nb_neurons, hidden_layers=hidden_layers, outermost_linear=True,
                 first_omega_0=3*kx, omega_0_t=3*wt)
    psi2 = Siren(in_features=3, out_features=1, hidden_features=nb_neurons, hidden_layers=2, outermost_linear=True,
                 first_omega_0=3*kx, omega_0_t=3*wt)
    psi3 = Siren(in_features=3, out_features=1, hidden_features=nb_neurons, hidden_layers=2, outermost_linear=True,
                 first_omega_0=3*kx, omega_0_t=3*wt)

    psi1.to(device)
    psi2.to(device)
    psi3.to(device)

    optim = torch.optim.Adam(lr=1e-4, params=psi1.parameters())

    pbar = tqdm(range(nb_epochs))

    loglossdata_1 = []
    loglossdyn_1 = []

    best = 1000

    for epoch in range(nb_epochs):
        for [[input_data, gt_psi1], [input_reg, _, _, forc]] in zip(dataloader1, regdataloader1):
            input_data, gt_psi1 = input_data.to(device), gt_psi1.to(device)
            output_data, _ = psi1(input_data)
            loss_data = ((output_data[:, 0].clone() - gt_psi1) ** 2).mean() # output_data if random obs else output_data[:, 0]

            input_reg, forc = input_reg.to(device), forc.to(device)
            output_reg, coords = psi1(input_reg)
            loss_dyn = lam*((dyn_psi1_0(std*output_reg, coords, forc) ** 2).mean())

            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_1.append(loss_data.cpu().detach().numpy())
            loglossdyn_1.append(loss_dyn.cpu().detach().numpy())

            if loss<=best:
                torch.save(psi1.state_dict(), "./Results/Psi_1_CBP_R1" + end)

        if epoch % 50 == 0:
            pbar.set_description("\n Total loss %0.6f. Progress " % loss)
            pbar.update(epoch)

    pbar.close()

    print("Psi_1 DONE")

    del data_psi1, regdata_psi1, dataloader1, regdataloader1

    data_psi2 = ImageRandomMask(img_size, t_size, 1, nb_int)
    regdata_psi2 = Regularizer(nb_reg, 1)

    coords = regdata_psi2.coords.to(device)
    output, coords = psi1(coords)
    gradpsi1 = gradient(std*output, coords)
    lappsi1 = divergence(gradpsi1, coords).detach()

    regdata_psi2.gradpsi_above = gradpsi1.detach()
    regdata_psi2.lappsi_above = lappsi1

    dataloader2 = DataLoader(data_psi2, batch_size=batch_size_int, pin_memory=False, num_workers=0, shuffle=True)
    regdataloader2 = DataLoader(regdata_psi2, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True)

    noise2 = data_psi2.noise
    idx2 = data_psi2.idx
    # torch.save(idx2, "./Results/obs_2"+end) # optional to show where obs are available

    optim = torch.optim.Adam(lr=1e-4, params=psi2.parameters())

    pbar = tqdm(range(nb_epochs))

    loglossdata_2 = []
    loglossdyn_2 = []

    best = 1000

    for epoch in pbar:
        for [[input_data, gt_psi2], [input_reg, gradpsi1, lappsi1, _, _]] in zip(dataloader2, regdataloader2):
            input_data, gt_psi2 = input_data.to(device), gt_psi2.to(device)
            output_data, _ = psi2(input_data)

            input_reg, gradpsi1, lappsi1 = input_reg.to(device), gradpsi1.to(device), lappsi1.to(device)
            output_reg, coords = psi2(input_reg)

            loss_data = ((output_data - gt_psi2) ** 2).mean()
            loss_dyn = lam*((dyn_psi2_0(std*output_reg, coords, gradpsi1, lappsi1) ** 2).mean())
            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_2.append(loss_data.cpu().detach().numpy())
            loglossdyn_2.append(loss_dyn.cpu().detach().numpy())

            if loss<=best:
                torch.save(psi2.state_dict(), "./Results/Psi_2_CBP_R1" + end)

        if epoch % 50 == 0:
            pbar.set_description("\n Total loss %0.6f. Progress " % loss)
            pbar.update(epoch)

    pbar.close()

    print("Psi_2 DONE")

    del data_psi2, regdata_psi2, dataloader2, regdataloader2

    data_psi3 = ImageRandomMask(img_size, t_size, 2, nb_bot)
    regdata_psi3 = Regularizer(nb_reg, 2)

    coords = regdata_psi3.coords.to(device)
    output, coords = psi2(coords)
    gradpsi2 = gradient(std*output, coords)
    lappsi2 = divergence(gradpsi2, coords).detach()

    regdata_psi3.gradpsi_above = gradpsi2.detach()
    regdata_psi3.lappsi_above = lappsi2

    dataloader3 = DataLoader(data_psi3, batch_size=batch_size_bot, pin_memory=False, num_workers=0, shuffle=True)
    regdataloader3 = DataLoader(regdata_psi3, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True)

    noise3 = data_psi3.noise
    idx3 = data_psi3.idx
    # torch.save(idx3, "./Results/obs_3"+end)

    optim = torch.optim.Adam(lr=1e-4, params=psi3.parameters())

    pbar = tqdm(range(nb_epochs))

    loglossdata_3 = []
    loglossdyn_3 = []

    best = 1000

    for epoch in pbar:
        for [[input_data, gt_psi3], [input_reg, gradpsi2, lappsi2]] in zip(dataloader3, regdataloader3):
            input_data, gt_psi3 = input_data.to(device), gt_psi3.to(device)
            output_data, _ = psi3(input_data)

            input_reg, gradpsi2, lappsi2 = input_reg.to(device), gradpsi2.to(device), lappsi2.to(device)
            output_reg, coords = psi3(input_reg)

            loss_data = ((output_data - gt_psi3) ** 2).mean()
            loss_dyn = lam*((dyn_psi3(std*output_reg, coords, gradpsi2, lappsi2) ** 2).mean())
            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_3.append(loss_data.cpu().detach().numpy())
            loglossdyn_3.append(loss_dyn.cpu().detach().numpy())

            if loss<=best:
                torch.save(psi3.state_dict(), "./Results/Psi_3_CBP_R1" + end)

        if epoch % 50 == 0:
            pbar.set_description("\n Total loss %0.6f. Progress " % loss)
            pbar.update(epoch)

    pbar.close()

    print("Psi_3 DONE")

    for round in range(nb_rounds-1):

        del data_psi3, regdata_psi3, dataloader3, regdataloader3

        print("ROUND %s"%(2+round))

        psi1_, coords1, mask, std = get_data()

        data_psi1 = SurfaceFitting(img_size, t_size, psi1_, coords1, mask, noise=noise1)
        # data_psi1 = ImageRandomMask(img_size, t_size, 0, nb_surface, idx=idx1)
        regdata_psi1 = Regularizer(nb_reg, 0)

        coords = regdata_psi1.coords.to(device)
        output, coords = psi2(coords)
        gradpsi2 = gradient(std*output, coords)
        lappsi2 = divergence(gradpsi2, coords).detach()

        regdata_psi1.gradpsi_below = gradpsi2.detach()
        regdata_psi1.lappsi_below = lappsi2

        dataloader1 = DataLoader(data_psi1, batch_size=batch_size_surf, pin_memory=False, num_workers=0, shuffle=True)
        regdataloader1 = DataLoader(regdata_psi1, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True)

        optim = torch.optim.Adam(lr=1e-4, params=psi1.parameters())

        pbar = tqdm(range(nb_epochs))

        best = 1000

        for epoch in pbar:
            for [[input_data, gt_psi1], [input_reg, gradpsi2, lappsi2, forc]] in zip(dataloader1, regdataloader1):
                input_data, gt_psi1 = input_data.to(device), gt_psi1.to(device)
                output_data, _ = psi1(input_data)
                loss_data = ((output_data[:, 0].clone() - gt_psi1) ** 2).mean() # output_data if random obs else output_data[:, 0]

                input_reg, gradpsi2, lappsi2, forc = input_reg.to(device), gradpsi2.to(device), lappsi2.to(device), forc.to(device)
                output_reg, coords = psi1(input_reg)
                loss_dyn = lam*((dyn_psi1(std*output_reg, coords, gradpsi2, lappsi2, forc) ** 2).mean())

                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_1.append(loss_data.cpu().detach().numpy())
                loglossdyn_1.append(loss_dyn.cpu().detach().numpy())

                if loss<=best:
                    torch.save(psi1.state_dict(), "./Results/Psi_1_CBP_R%s"%(round+2) + end)

            if epoch % 50 == 0:
                pbar.set_description("\n Total loss %0.6f. Progress " % loss)
                pbar.update(epoch)

        pbar.close()

        print("Psi_1 DONE")

        del data_psi1, regdata_psi1, dataloader1, regdataloader1

        data_psi2 = ImageRandomMask(img_size, t_size, 1, nb_int, idx=idx2, noise=noise2)
        regdata_psi2 = Regularizer(nb_reg, 1)

        coords = regdata_psi2.coords.to(device)

        output, coords1 = psi1(coords)
        gradpsi1 = gradient(std*output, coords1)
        lappsi1 = divergence(gradpsi1, coords1).detach()

        regdata_psi2.gradpsi_above = gradpsi1.detach()
        regdata_psi2.lappsi_above = lappsi1

        output, coords2 = psi3(coords)
        gradpsi3 = gradient(std*output, coords2)
        lappsi3 = divergence(gradpsi3, coords2).detach()

        regdata_psi2.gradpsi_below = gradpsi3.detach()
        regdata_psi2.lappsi_below = lappsi3

        dataloader2 = DataLoader(data_psi2, batch_size=batch_size_int, pin_memory=False, num_workers=0, shuffle=True)
        regdataloader2 = DataLoader(regdata_psi2, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True)

        optim = torch.optim.Adam(lr=1e-4, params=psi2.parameters())

        pbar = tqdm(range(nb_epochs))

        best = 1000

        for epoch in pbar:
            for [[input_data, gt_psi2], [input_reg, gradpsi1, lappsi1, gradpsi3, lappsi3]] in zip(dataloader2, regdataloader2):
                input_data, gt_psi2 = input_data.to(device), gt_psi2.to(device)
                output_data, _ = psi2(input_data)

                input_reg, gradpsi1, lappsi1 = input_reg.to(device), gradpsi1.to(device), lappsi1.to(device)
                gradpsi3, lappsi3 = gradpsi3.to(device), lappsi3.to(device)
                output_reg, coords = psi2(input_reg)

                loss_data = ((output_data - gt_psi2) ** 2).mean()
                loss_dyn = lam*((dyn_psi2(std*output_reg, coords, gradpsi1, gradpsi3, lappsi1, lappsi3) ** 2).mean())
                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_2.append(loss_data.cpu().detach().numpy())
                loglossdyn_2.append(loss_dyn.cpu().detach().numpy())

                if loss<=best:
                    torch.save(psi2.state_dict(), "./Results/Psi_2_CBP_R%s"%(round+2) + end)

            if epoch % 50 == 0:
                pbar.set_description("\n Total loss %0.6f. Progress " % loss)
                pbar.update(epoch)

        pbar.close()

        print("Psi_2 DONE")

        del data_psi2, regdata_psi2, dataloader2, regdataloader2

        data_psi3 = ImageRandomMask(img_size, t_size, 2, nb_bot, idx=idx3, noise=noise3)
        regdata_psi3 = Regularizer(nb_reg, 2)

        coords = regdata_psi3.coords.to(device)
        output, coords = psi2(coords)
        gradpsi2 = gradient(std*output, coords)
        lappsi2 = divergence(gradpsi2, coords).detach()

        regdata_psi3.gradpsi_above = gradpsi2.detach()
        regdata_psi3.lappsi_above = lappsi2

        dataloader3 = DataLoader(data_psi3, batch_size=batch_size_bot, pin_memory=False, num_workers=0, shuffle=True)
        regdataloader3 = DataLoader(regdata_psi3, batch_size=batch_size_reg, pin_memory=False, num_workers=0, shuffle=True)

        optim = torch.optim.Adam(lr=1e-4, params=psi3.parameters())

        pbar = tqdm(range(nb_epochs))

        best = 1000

        for epoch in pbar:
            for [[input_data, gt_psi3], [input_reg, gradpsi2, lappsi2]] in zip(dataloader3, regdataloader3):
                input_data, gt_psi3 = input_data.to(device), gt_psi3.to(device)
                output_data, _ = psi3(input_data)

                input_reg, gradpsi2, lappsi2 = input_reg.to(device), gradpsi2.to(device), lappsi2.to(device)
                output_reg, coords = psi3(input_reg)

                loss_data = ((output_data - gt_psi3) ** 2).mean()
                loss_dyn = lam*((dyn_psi3(std*output_reg, coords, gradpsi2, lappsi2) ** 2).mean())
                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_3.append(loss_data.cpu().detach().numpy())
                loglossdyn_3.append(loss_dyn.cpu().detach().numpy())

                if loss<=best:
                    torch.save(psi3.state_dict(), "./Results/Psi_3_CBP_R%s"%(round+2) + end)

            if epoch % 50 == 0:
                pbar.set_description("\n Total loss %0.6f. Progress " % loss)
                pbar.update(epoch)

        pbar.close()

        print("Psi_3 DONE")

    del data_psi3, regdata_psi3, dataloader3, regdataloader3

    print("")
    print("lam=%s DONE"%lam)
    print("")
    losses_1 = np.stack((loglossdata_1, loglossdyn_1))
    losses_23 = np.stack((loglossdata_2, loglossdyn_2,
                loglossdata_3, loglossdyn_3))

    np.save("./Results/Losses_CBP_3L_1"+end, losses_1)
    np.save("./Results/Losses_CBP_3L_23"+end, losses_23)