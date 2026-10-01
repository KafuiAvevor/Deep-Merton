Deep Solver for Merton's Optimal Control

This repository implements a custom hybrid deep learning solver for Merton's Portfolio Optimization problem, bridging **Forward-Backward Stochastic Differential Equations (FBSDEs)** with **Physics-Informed Neural Networks (PINNs)**.

Unlike standard approaches that rely purely on terminal loss, this solver enforces the Hamilton-Jacobi-Bellman (HJB) algebraic constraints and the Martingale Optimality conditions locally along forward-simulated stochastic trajectories.

## The Mathematical Framework

Consider an agent allocating wealth $X_t$ between a risky asset and a risk-free asset to maximize expected CRRA utility:

$$J(X_0) = \mathbb{E}\\left[\\frac{X_T^{1-\\gamma}}{1-\\gamma}\\right]$$

The portfolio wealth evolves according to the stochastic differential equation:

$$dX_t = \\mu\\pi_t X_tdt + \\sigma\\pi_tX_t dW_t +r[1-\\pi_t] X_tdt$$

where $\\pi_t$ is the proportion of wealth invested in the risky asset.

### The FBSDE Formulation
Let $Y_t$ be the dynamic value function $H(t,x)$.

$$H(t,X_t) = \sup_\pi\mathbb{E}\\left[\\frac{X_T^{1-\\gamma}}{1-\\gamma}\bigm| \mathcal{F}_t\\right]$$

Applying Ito's Lemma, the process $Y_t$ evolves according to the process:

$$ dY_t = \left[\underbrace{\partial_t H + rx\partial_x H +(\mu - r)\pi_t x\partial_x H +\frac{1}{2} \sigma^2 \pi^2 x^2 \partial_{xx} H}_{\mathcal{D}^{\pi}}\right] dt + \underbrace{\sigma \pi x \partial_x H}_{Z_t} dW_t, \quad  Y_T = \frac{X_T^{1-\gamma}}{1-\gamma} $$

or , in terms of Z:

$$ dY_t =  \left[\underbrace{\\partial_tY_t + r(1-\\gamma)Y_t + \\frac{\\mu -r}{\\sigma}Z_t - \\frac{\\gamma}{2(1-\\gamma)}\\frac{Z_t^2}{Y_t}}_{\mathcal{D}^{Z}}\right] dt +Z_t dW_t, , \quad  Y_T = \frac{X_T^{1-\gamma}}{1-\gamma}$$

The Martingale Optimality Principle states that under the optimal control policy $Z^*$:
1. The drift of the value process must be zero: $\\mathcal{D}(Z^*) = 0 $ 
2. The supremum is attained at the stationary point: $\\partial_Z\\mathcal{D} = 0$

## Deep Learning Methodology

This solver parameterizes the value function and the control process using two distinct PyTorch neural networks:
*   `YNet(t, X)`: Approximates the dynamic value function $Y_t$.
*   `ZNet(t, X)`: Approximates the control-related process $Z_t$.

### The Loss Function
Instead of relying solely on the terminal condition (which provides a weak learning signal for complex control problems), the loss function explicitly embeds the optimality conditions across the entire state-time trajectory.

The total loss is a dynamically weighted sum of three components:

1.  **Terminal Loss:** Ensures the value function matches the utility payoff at maturity $T$.

   $$\\mathcal{L}_{terminal} = \\mathbb{E}\\left[\\left(Y_T - \\frac{X_T^{1-\\gamma}}{1-\\gamma}\\right)^2\\right]$$

3.  **PDE Loss (HJB Residual):** Enforces the condition that the drift $\\mathcal{D}$ must be zero under optimal control. We compute exact spatial and temporal derivatives using PyTorch's `autograd`.

   $$\\mathcal{L}_{PDE} = \\mathbb{E}[\\mathcal{D}^2]$$

5.  **First-Order Optimality Loss:** Explicitly forces the network to satisfy the mathematical definition of a local maximum along the trajectory.

   $$\\mathcal{L}_{deriv} = \\mathbb{E}\\left[\\left(\\frac{\\partial \\mathcal{D}}{\\partial Z}\\right)^2\\right]$$

   
