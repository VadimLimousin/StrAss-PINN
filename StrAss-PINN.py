try:
    from utils import *

    from netCDF4 import Dataset as d

    import torch
    from torch.utils.data import DataLoader, Dataset

    from PIL import Image
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    import numpy as np

    from time import time
    import os
    import argparse
except ImportError as e:
    print(f"An error occurred: {e}. Please make sure all required packages are installed.", flush=True)

def maybe_save(params, epoch, psi, level):
    if epoch % params.save_every == 0:
        torch.save(psi.state_dict(), "." + params.home + f"/Psi_{level+1}_CBP_R{params.round}" + end)

def maybe_log(params, epoch, psi, level, loss):
    levels = ["surface", "interior", "abyss"]
    if epoch % params.log_every == 0:
        valloss = quick_val(params, psi, level, (51, 10), plot=epoch%(10*params.log_every) == 0)
        if epoch % params.print_every == 0:
            print("Epoch %s. Validation loss: %0.6f." % (epoch, valloss), flush=True)
            print("Total loss: %0.6f." % loss.detach(), flush=True)

        if params.use_wandb:
            run.log({f"Val loss R{params.round} "+levels[level]: valloss, f"Train loss R{params.round} "+levels[level]: loss.detach()})

parser = argparse.ArgumentParser(description="Chose config.")
parser.add_argument('--config', type=str, default="ARGO")
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

path_config = "./Codes/utils/Config.yaml"
params = YParams(os.path.abspath(path_config), args.config, print_params=True)
params["device"] = device

for lam in params.LAM:
    psi1 = Siren(in_features=3, out_features=1, hidden_features=params.nb_neurons,
                 hidden_layers=params.hidden_layers, outermost_linear=True,
                 first_omega_0=3 * params.kx, omega_0_t=3 * params.wt)
    psi2 = Siren(in_features=3, out_features=1, hidden_features=params.nb_neurons,
                 hidden_layers=params.hidden_layers, outermost_linear=True,
                 first_omega_0=3 * params.kx, omega_0_t=3 * params.wt)
    psi3 = Siren(in_features=3, out_features=1, hidden_features=params.nb_neurons,
                 hidden_layers=params.hidden_layers, outermost_linear=True,
                 first_omega_0=3 * params.kx, omega_0_t=3 * params.wt)

    if params.use_wandb:
        run = log_wandb(params, lam)
        params["run"] = run

    end = "_Strat-PINN_lam=%s" % lam
    params["end"] = end
    params["round"] = 1

    dataloader1, noise1 = get_dataloader_surf(params)

    psi1.to(device)
    psi2.to(device)
    psi3.to(device)

    optim = torch.optim.Adam(lr=params.lr, params=psi1.parameters())
    loglossdata_1 = []
    loglossdyn_1 = []
    user_time = time()

    for epoch in range(params.nb_epochs):
        for input_data, gt_psi1 in dataloader1:
            input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)
            forc = wind_forc(input_reg)

            input_data, gt_psi1 = input_data.to(device), gt_psi1.to(device)
            output_data, _ = psi1(input_data)
            loss_data = ((output_data[:, 0].clone() - gt_psi1) ** 2).mean() # output_data if random obs else output_data[:, 0]

            output_reg, coords = psi1(input_reg)
            loss_dyn = lam*((dyn_psi1_0(params.std*output_reg, coords, forc) ** 2).mean())

            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_1.append(loss_data.cpu().detach().numpy())
            loglossdyn_1.append(loss_dyn.cpu().detach().numpy())

        maybe_save(params, epoch, psi1, 0)
        maybe_log(params, epoch, psi1, 0, loss)

    torch.save(psi1.state_dict(), "./"+params.home+"/Psi_1_CBP_R1" + end)
    print(f"\n Psi_1 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

    del dataloader1

    dataloader2, noise2, idx2 = get_dataloader_int(params, psi1)

    optim = torch.optim.Adam(lr=params.lr, params=psi2.parameters())

    loglossdata_2 = []
    loglossdyn_2 = []
    user_time = time()

    for epoch in range(params.nb_epochs):
        for input_data, gt_psi2 in dataloader2:
            input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)

            output_surf, coords1 = psi1(input_reg)
            gradpsi1 = gradient(output_surf, coords1)
            lappsi1 = divergence(gradpsi1, coords1).detach()
            gradpsi1 = gradpsi1.detach()

            input_data, gt_psi2 = input_data.to(device), gt_psi2.to(device)
            output_data, _ = psi2(input_data)
            output_reg, coords = psi2(input_reg)

            loss_data = ((output_data - gt_psi2) ** 2).mean()
            loss_dyn = lam*((dyn_psi2_0(params.std*output_reg, coords, gradpsi1, lappsi1) ** 2).mean())
            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_2.append(loss_data.cpu().detach().numpy())
            loglossdyn_2.append(loss_dyn.cpu().detach().numpy())

        maybe_save(params, epoch, psi2, 1)
        maybe_log(params, epoch, psi2, 1, loss)

    print(f"\n Psi_2 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

    torch.save(psi2.state_dict(), "."+params.home+"/Psi_2_CBP_R1" + end)

    del dataloader2

    dataloader3, noise3, idx3 = get_dataloader_bot(params, psi2)

    optim = torch.optim.Adam(lr=params.lr, params=psi3.parameters())

    loglossdata_3 = []
    loglossdyn_3 = []
    user_time = time()

    for epoch in range(params.nb_epochs):
        for input_data, gt_psi3 in dataloader3:
            input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)

            output_int, coords2 = psi2(input_reg)
            gradpsi2 = gradient(output_int, coords2)
            lappsi2 = divergence(gradpsi2, coords2).detach()
            gradpsi2 = gradpsi2.detach()

            input_data, gt_psi3 = input_data.to(device), gt_psi3.to(device)
            output_data, _ = psi3(input_data)
            output_reg, coords = psi3(input_reg)

            loss_data = ((output_data - gt_psi3) ** 2).mean()
            loss_dyn = lam*((dyn_psi3(params.std*output_reg, coords, gradpsi2, lappsi2) ** 2).mean())
            loss = loss_data + loss_dyn

            optim.zero_grad()
            loss.backward()
            optim.step()

            loglossdata_3.append(loss_data.cpu().detach().numpy())
            loglossdyn_3.append(loss_dyn.cpu().detach().numpy())

        maybe_save(params, epoch, psi3, 2)
        maybe_log(params, epoch, psi3, 2, loss)

    print(f"\n Psi_3 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

    torch.save(psi3.state_dict(), "."+params.home+"/Psi_3_CBP_R1" + end)

    params["batch_size_reg"] = 70*params.batch_size_reg//100
    params["nb_reg"] = 13 * params.batch_size_reg

    for r in range(params.nb_rounds-1):
        del dataloader3
        params["round"] = r+2
        print(f"ROUND {r+2}", flush=True)

        dataloader1 = get_dataloader_surf(params, noise=noise1)

        optim = torch.optim.Adam(lr=params.lr, params=psi1.parameters())
        user_time = time()

        for epoch in range(params.nb_epochs):
            for input_data, gt_psi1 in dataloader1:
                input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)
                forc = wind_forc(input_reg)

                output_int, coords2 = psi2(input_reg)
                gradpsi2 = gradient(output_int, coords2)
                lappsi2 = divergence(gradpsi2, coords2).detach()
                gradpsi2 = gradpsi2.detach()

                input_data, gt_psi1 = input_data.to(device), gt_psi1.to(device)
                output_data, _ = psi1(input_data)
                output_reg, coords = psi1(input_reg)

                loss_data = ((output_data[:,0].clone() - gt_psi1) ** 2).mean()
                loss_dyn = lam*((dyn_psi1(params.std*output_reg, coords, gradpsi2, lappsi2, forc) ** 2).mean())

                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_1.append(loss_data.cpu().detach().numpy())
                loglossdyn_1.append(loss_dyn.cpu().detach().numpy())

            maybe_save(params, epoch, psi1, 0)
            maybe_log(params, epoch, psi1, 0, loss)

        print(f"\n Psi_1 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

        torch.save(psi1.state_dict(), "."+params.home+f"/Psi_1_CBP_R{r+2}" + end)

        del dataloader1

        dataloader2 = get_dataloader_int(params, psi1, noise=noise2, idx=idx2)

        optim = torch.optim.Adam(lr=params.lr, params=psi2.parameters())
        user_time = time()

        for epoch in range(params.nb_epochs):
            for input_data, gt_psi2 in dataloader2:
                input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)

                output_surf, coords1 = psi1(input_reg)
                gradpsi1 = gradient(output_surf, coords1)
                lappsi1 = divergence(gradpsi1, coords1).detach()
                gradpsi1 = gradpsi1.detach()

                output_bot, coords3 = psi3(input_reg)
                gradpsi3 = gradient(output_bot, coords3)
                lappsi3 = divergence(gradpsi3, coords3).detach()
                gradpsi3 = gradpsi3.detach()

                input_data, gt_psi2 = input_data.to(device), gt_psi2.to(device)
                output_data, _ = psi2(input_data)
                output_reg, coords = psi2(input_reg)

                loss_data = ((output_data - gt_psi2) ** 2).mean()
                loss_dyn = lam*((dyn_psi2(params.std*output_reg, coords, gradpsi1, gradpsi3, lappsi1, lappsi3) ** 2).mean())
                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_2.append(loss_data.cpu().detach().numpy())
                loglossdyn_2.append(loss_dyn.cpu().detach().numpy())

            maybe_save(params, epoch, psi2, 1)
            maybe_log(params, epoch, psi2, 1, loss)

        print(f"\n Psi_2 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

        torch.save(psi2.state_dict(), "."+params.home+f"/Psi_2_CBP_R{r+2}" + end)

        del dataloader2

        dataloader3 = get_dataloader_bot(params, psi2, noise=noise3, idx=idx3)

        optim = torch.optim.Adam(lr=params.lr, params=psi3.parameters())
        user_time = time()

        for epoch in range(params.nb_epochs):
            for input_data, gt_psi3 in dataloader3:
                input_reg = torch.zeros(params.batch_size_reg, 3).uniform_(-1, 1).to(device)

                output_int, coords2 = psi2(input_reg)
                gradpsi2 = gradient(output_int, coords2)
                lappsi2 = divergence(gradpsi2, coords2).detach()
                gradpsi2 = gradpsi2.detach()

                input_data, gt_psi3 = input_data.to(device), gt_psi3.to(device)
                output_data, _ = psi3(input_data)
                output_reg, coords = psi3(input_reg)

                loss_data = ((output_data - gt_psi3) ** 2).mean()
                loss_dyn = lam*((dyn_psi3(params.std*output_reg, coords, gradpsi2, lappsi2) ** 2).mean())
                loss = loss_data + loss_dyn

                optim.zero_grad()
                loss.backward()
                optim.step()

                loglossdata_3.append(loss_data.cpu().detach().numpy())
                loglossdyn_3.append(loss_dyn.cpu().detach().numpy())

            maybe_save(params, epoch, psi3, 2)
            maybe_log(params, epoch, psi3, 2, loss)

        print(f"\n Psi_3 DONE in {int((time() - user_time)/60)} minutes \n", flush=True)

    torch.save(psi3.state_dict(), "."+params.home+f"/Psi_3_CBP_R{r+2}" + end)

    run.finish()

    print("")
    print("lam=%s DONE"%lam)
    print("")
    losses_1 = np.stack((loglossdata_1, loglossdyn_1))
    losses_23 = np.stack((loglossdata_2, loglossdyn_2,
                loglossdata_3, loglossdyn_3))

    np.save("."+params.home+"/Losses_CBP_3L_1"+end, losses_1)
    np.save("."+params.home+"/Losses_CBP_3L_23"+end, losses_23)