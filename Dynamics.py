import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2

rx = 2/4000e3
rt = 2/86400/10/10

H1 = torch.tensor(350)
H2 = torch.tensor(750)
H3 = torch.tensor(2900)
g1 = 4.54545455e-05*(H1 + H2)/2
g2 = 6.84931507e-06*(H2 + H3)/2
f0 = 9.4e-5

L1 = torch.sqrt(H1*g1)/f0 # rayons de déformation de Rossby
L2 = torch.sqrt(H2*g2)/f0
L12 = torch.sqrt(H2*g1)/f0
L3 = torch.sqrt(H3*g2)/f0
beta = 1.7e-11
tau0 = 1.3e-5
nu = 40
rE = f0*2/2/H3

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(1, y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def total_deriv(y, x, gradpsi):
    grad = gradient(y, x)
    return rt/rx**2*grad[:, 0].clone() - gradpsi[:, 2].clone() * grad[:, 1].clone() + gradpsi[:, 1].clone() * grad[:, 2].clone()

def discrete_total_deriv(y, gradpsi):
    return rt/rx**2*y[:, 0].clone() - gradpsi[:, 2].clone() * y[:, 1].clone() + gradpsi[:, 1].clone() * y[:, 2].clone()

def wind_forc(coords):
    return - tau0 * torch.pi * torch.sin(torch.pi * coords[:, 2].clone())

def dyn_psi1(psi1, coords, gradpsi2, lappsi2, forc):
    gradpsi1 = gradient(psi1, coords)

    z1 = laplace(psi1, coords)
    t1 = rx**2*total_deriv(z1, coords, gradpsi1) - 1/L1**2 * gradpsi1[:, 0].clone()
    t2 = discrete_total_deriv(gradpsi2 / L1 ** 2, gradpsi1)
    t3 = beta * gradpsi1[:, 1].clone()
    t4 = nu * (rx**2*laplace(z1, coords) - 1/L1**2*(z1 - lappsi2))[:, 0].clone()
    return t1 + t2 + t3/rx + forc + t4

def dyn_psi1_0(psi1, coords, forc):
    gradpsi1 = gradient(psi1, coords)

    z1 = laplace(psi1, coords)
    t1 = rx**2*total_deriv(z1, coords, gradpsi1) - 1/L1**2 * gradpsi1[:, 0].clone()
    t2 = beta * gradpsi1[:, 1].clone()
    t3 = nu * (rx**2*laplace(z1, coords) - 1/L1**2*z1)
    return t1 + t2/rx + forc + t3[:, 0].clone()

def dyn_psi2(psi2, coords, gradpsi1, gradpsi3, lappsi1, lappsi3):
    gradpsi2 = gradient(psi2, coords)

    z2 = laplace(psi2, coords)
    t1 = rx**2*total_deriv(z2, coords, gradpsi2) - (1 / L12**2 + 1 / L2**2) * gradpsi2[:, 0].clone()
    t2 = discrete_total_deriv(gradpsi1 / L12**2 + gradpsi3 / L2**2, gradpsi2)
    t3 = beta * gradpsi2[:, 1].clone()
    t4 = nu * (rx**2*laplace(z2, coords) - (1/L12**2 + 1/L2**2)*z2 + 1/L12**2*lappsi1 + 1/L2**2*lappsi3)
    return t1 + t2 + t3/rx + t4[:, 0].clone()

def dyn_psi2_0(psi2, coords, gradpsi1, lappsi1):
    gradpsi2 = gradient(psi2, coords)

    z2 = laplace(psi2, coords)
    t1 = rx**2*total_deriv(z2, coords, gradpsi2) - (1 / L12**2 + 1 / L2**2) * gradpsi2[:, 0].clone()
    t2 = discrete_total_deriv(gradpsi1 / L12**2, gradpsi2)
    t3 = beta * gradpsi2[:, 1].clone()
    t4 = nu * (rx**2*laplace(z2, coords) - (1/L12**2 + 1/L2**2)*z2 + 1/L12**2*lappsi1)
    return t1 + t2 + t3/rx + t4[:, 0].clone()

def dyn_psi3(psi3, coords, gradpsi2, lappsi2):
    gradpsi3 = gradient(psi3, coords)

    z3 = laplace(psi3, coords)
    t1 = rx**2*total_deriv(z3, coords, gradpsi3) - 1 / L3**2 * gradpsi3[:, 0].clone()
    t2 = discrete_total_deriv(gradpsi2 / L3**2, gradpsi3)
    t3 = beta * gradpsi3[:, 1].clone()
    t4 = nu * (rx**2*laplace(z3, coords) - 1/L3**2*(z3 - lappsi2))
    bot = -rE * z3
    return t1 + t2 + t3/rx + t4[:, 0].clone() + bot[:, 0].clone()