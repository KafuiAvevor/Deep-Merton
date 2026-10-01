import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

T = 1.0
N = 50
dt = T / N
sqrt_dt = dt**0.5
mu = 0.08
r = 0.02
sigma = 0.2
gamma = 5
x0 = 1.0
alpha = 0.02

a = (1 - gamma) * (r + (mu - r)**2 / (2 * gamma * sigma**2))
pi_known = (mu - r) / (gamma * sigma**2)  
true_Y0 = np.exp(a * T) * (x0**(1 - gamma)) / (1 - gamma)
true_aY0 = -a * true_Y0
z_star = (mu - r)/sigma * (1 - gamma) / gamma * true_Y0

device = "cpu"

class YNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        with torch.no_grad():
            self.net[-1].bias.fill_(x0**(1 - gamma) / (1 - gamma))
            self.net[-1].weight.mul_(0.1)

    def forward(self, t, x):
        inp = torch.cat([t / T, x / x0], dim=1)
        return self.net(inp)
class ZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t, x):
        inp = torch.cat([t / T, x / x0], dim=1)
        return self.net(inp)

z_net = ZNet().to(device)
y_net = YNet().to(device)
optimizer = optim.Adam(list(y_net.parameters()) + list(z_net.parameters()), lr=1e-4)   

n_epochs = 5000
batch_size = 256

pi_history = []
Y0_hist = []
Z_history = []
dtY_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    dW = torch.randn(batch_size, N, 1, device=device) * sqrt_dt
    X = torch.full((batch_size, 1), x0, device=device)
    pi_epoch = None
    Y0_epoch = None
    Z_epoch = []
    dtY_epoch = []
    f_loss_total = 0
    derivative_loss_total = 0

    for n in range(N):
        t = torch.full((batch_size, 1), n * dt, device=device, requires_grad=True)
        Xg = X.detach().requires_grad_(True)
        H = y_net(t, Xg)
        Z = z_net(t, Xg)
        Ht = torch.autograd.grad(H.sum(), t, create_graph=True)[0]
        pi = Z / (sigma * (1 - gamma) * H) 
        
        Z_epoch.append(Z.mean().detach().item())
        dtY_epoch.append(Ht.mean().detach().item())

        if n == 0:
            pi_epoch = pi.mean().detach().item()
            Y0_epoch = H.mean().detach().item()

        f = -(Ht
         + r * (1 - gamma) * H
            + (mu - r) / sigma * Z
            - gamma / (2*(1 - gamma)) * (Z**2) / (H ))
        df_dz = torch.autograd.grad(
            f.sum(),
            Z,
            create_graph=True
            )[0]
        f_loss_total += torch.mean(f**2)
        derivative_loss_total += torch.mean(df_dz**2)
        pid = pi.detach().clamp(-20, 20)   
        X = X * (1 + r*dt + pid*(mu - r)*dt + pid*sigma*dW[:, n])
        X = X.clamp(min=1e-4)             

    tT = torch.full((batch_size, 1), T, device=device)
    Y_T = y_net(tT, X)
    Y_T_target = X**(1 - gamma) / (1 - gamma)
    terminal_loss = torch.mean((Y_T - Y_T_target)**2)

    if epoch < 200:
        pde_w = 0.0
        deriv_w = 1.0
    elif f_loss_total.item() > 1e-3 or derivative_loss_total.item() > 1e-3: 
        pde_w = 5.0
        deriv_w = 2.0
    else:
        pde_w = 0.1
        deriv_w = 0.1
    loss = (
    10.0 * terminal_loss
    + pde_w * f_loss_total
    + deriv_w * derivative_loss_total
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(y_net.parameters(), max_norm=10.0)
    optimizer.step()

    Y0_hist.append(Y0_epoch)
    pi_history.append(pi_epoch)
    Z_curr = float(sum(Z_epoch) / len(Z_epoch))
    Z_history.append(Z_curr)
    dtY_curr = float(sum(dtY_epoch) / len(dtY_epoch))
    dtY_history.append(dtY_curr)


    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}, PDE loss: {f_loss_total.item():.6e}, Derivative loss: {derivative_loss_total.item():.6e}, "
              f"Terminal loss: {terminal_loss.item():.6e}, Y0: {Y0_epoch:.4f}, "
              f"pi: {pi_epoch:.4f} (known: {pi_known:.4f}), avg Z: {Z_curr:.4f}, "
              f"avg dtY: {dtY_curr:.4f} "
              f"a*Y0 (known: {true_aY0:.4f}): {-a*Y0_epoch:.4f}")

    if (epoch >= 200
    and f_loss_total < 1e-3
    and derivative_loss_total < 1e-3
    and terminal_loss < 2e-3
    and abs(pi_epoch - pi_history[-50]) < 5e-4):
        print(f"Converged at epoch {epoch}")
        break

pi_opt = pi_epoch
Y0_opt = Y0_epoch
dtY_opt = dtY_curr

