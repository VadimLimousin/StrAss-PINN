import torch
from torch.utils.data import Dataset, DataLoader
from .utils import get_flow_tensor, get_grid, get_data, read_data
from .Dynamics import wind_forc

class SurfaceFitting(Dataset):
    def __init__(self, params, sidelen, timelen, psi, coords, mask, noise=None):
        super().__init__()
        self.shape = (timelen, sidelen, sidelen)
        self.coords = coords
        self.mask = mask
        self.noise = params.noise_strength*torch.randn(psi.shape) if noise is None else noise
        self.psi = psi if self.noise is None else psi + self.noise

    def __len__(self):
        return len(self.psi)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError
        return self.coords[idx], self.psi[idx]

class ImageRandomMask(Dataset):
    """
        Generates the dataset of pseudo-observations. nb_buoys points are drawn uniformly on the grid every day (or every 10 days for the abyss).
    """
    def __init__(self, params, sidelen, timelen, level, nb_points, Psi, idx=None, noise=None):
        if idx is None:
            self.idx = torch.zeros(nb_points, dtype=torch.int)
            t_max = nb_points//params.nb_buoys
            for t in range(0, t_max):
                self.idx[t*params.nb_buoys:(t+1)*params.nb_buoys] = sidelen*sidelen*t*timelen//t_max + torch.randperm(sidelen*sidelen)[:params.nb_buoys]
        else:
            self.idx = idx
        super().__init__()
        psi = get_flow_tensor(sidelen, timelen, level, params.mean, params.std).view(-1, 1)[self.idx]
        psi_above = get_flow_tensor(sidelen, timelen, level, params.mean, params.std).view(-1, 1)[self.idx]
        self.noise = params.noise_strength*torch.randn(psi.shape) if noise is None else noise
        self.coords = get_grid(sidelen, timelen)[self.idx, :]

        # The buoy measures psi_{l+1} - psi_l to which we add the estimate Psi_l and possibly noise
        self.psi = psi - psi_above + Psi(self.coords).detach() if self.noise is None else psi + self.noise

    def __len__(self):
        return len(self.psi)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError

        return self.coords[idx], self.psi[idx]

class StaticRandomMask(Dataset):
    """
        Testing the case when buoys do not move for a given duration sampling_every.
    """
    def __init__(self, params, sidelen, timelen, level, nb_points, Psi, idx=None, noise=None, sample_every=None):
        if idx is None:
            t_max = nb_points // params.nb_buoys

            sample_every = sample_every if sample_every is not None else t_max
            sampling_times = list(range(0, t_max, sample_every))

            self.idx = torch.zeros(nb_points, dtype=torch.int)
            for t in range(0, t_max):
                if t in sampling_times:
                    snap = torch.randperm(sidelen * sidelen)[:params.nb_buoys]

                self.idx[t*params.nb_buoys:(t+1)*params.nb_buoys] = sidelen*sidelen*t*timelen//t_max + snap
        else:
            self.idx = idx
        super().__init__()
        psi = get_flow_tensor(sidelen, timelen, level, params.mean, params.std).view(-1, 1)[self.idx]
        psi_above = get_flow_tensor(sidelen, timelen, level, params.mean, params.std).view(-1, 1)[self.idx]
        self.noise = params.noise_strength*torch.randn(psi.shape) if noise is None else noise
        self.coords = get_grid(sidelen, timelen)[self.idx, :]

        # The buoy measures psi_{l+1} - psi_l to which we add the estimate Psi_l and possibly noise
        self.psi = psi - psi_above + Psi(self.coords).detach() if self.noise is None else psi + self.noise

    def __len__(self):
        return len(self.psi)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError

        return self.coords[idx], self.psi[idx]

class Regularizer(Dataset):
    def __init__(self, nb_reg, level):
        super().__init__()
        self.coords = torch.zeros(nb_reg, 3).uniform_(-1, 1) # TESTER AVEC LA GRILLE
        self.level = level
        if self.level==0:
            self.forc = wind_forc(self.coords)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if idx > self.__len__(): raise IndexError

        if self.level==0:
            return self.coords[idx], self.forc[idx]
        elif self.level==1:
            return self.coords[idx]
        else:
            return self.coords[idx]

def get_dataloader_surf(params, noise=None, gd=None):
    read_data()

    if gd is None:
        psi1, coords1, mask, std = get_data()
    else:
        psi1, coords1, mask, std = gd()

    data_psi1 = SurfaceFitting(params, params.img_size, params.t_size, psi1, coords1, mask, noise=noise)


    dataloader1 = DataLoader(data_psi1, batch_size=params.batch_size_surf, pin_memory=torch.cuda.is_available(),
                             num_workers=params.num_workers, shuffle=True, drop_last=True)  # observation points

    if params.round>1:
        return dataloader1
    else:
        return dataloader1, data_psi1.noise

def get_dataloader_int(params, psi1, noise=None, idx=None):
    data_psi2 = StaticRandomMask(params, params.img_size, params.t_size, 1, params.nb_int, psi1,
                                 noise=noise, idx=idx, sample_every=params.timescale)

    dataloader2 = DataLoader(data_psi2, batch_size=params.batch_size_int, pin_memory=torch.cuda.is_available(),
                             num_workers=params.num_workers, shuffle=True, drop_last=True)

    noise2 = data_psi2.noise
    idx2 = data_psi2.idx
    torch.save(idx2, "." + params.home + "/obs_2" + params.end) # optional to show where obs are available

    if params.round>1:
        return dataloader2
    else:
        return dataloader2, noise2, idx2

def get_dataloader_bot(params, psi2, noise=None, idx=None):
    data_psi3 = StaticRandomMask(params, params.img_size, params.t_size, 2, params.nb_bot, psi2,
                                 noise=noise, idx=idx, sample_every=params.timescale)

    dataloader3 = DataLoader(data_psi3, batch_size=params.batch_size_bot, pin_memory=torch.cuda.is_available(),
                             num_workers=params.num_workers, shuffle=True, drop_last=True)

    noise3 = data_psi3.noise
    idx3 = data_psi3.idx
    torch.save(idx3, "." + params.home + "/obs_3" + params.end)

    if params.round>1:
        return dataloader3
    else:
        return dataloader3, noise3, idx3