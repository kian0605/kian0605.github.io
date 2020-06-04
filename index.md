###### tags: `deep-learning`
# The note of "Adversarial GMM learning"


* $h$ refers to as a parametric function consisting of parameters denoted by $\boldsymbol{\theta}$, e.g. sigmoid function, $y_i=\frac{\theta_1}{1+\exp{(-\theta_2w_i)}}+\epsilon_i$, and $\theta_1=\theta_2=2$, that is $\rho(\mathbf{z}_i;h)$.
* $f$ denotes the function within the function class $\mathcal{F}$, and we consider multiple functions (known) with different probabilities~($\sigma_f$) to characterize the mixed strategy taken from adversary.
* Therefore, the main loss function (empirical version) is defined as follows
\[
L(h,\boldsymbol{\sigma}) = \sum_{f\in\mathcal{F}}\sigma_f\left[\frac{1}{N}\sum_{i=1}^N(\rho(\mathbf{z}_i;h)f(x_i))\right]^2,\quad i=1,...,N
\]
where $\boldsymbol{\sigma}=(\sigma_1,...,\sigma_K)^{\mathrm{T}}$, $\boldsymbol{z}_i=(y_i,w_i)^{\mathrm{T}}$, and $K$ denotes the number of functions considered. Accordingly, we can derive the gradient of the loss function.
\[
 \nabla_{\boldsymbol{\theta}}L(h,\boldsymbol{\sigma})=\sum_{f\in\mathcal{F}}\sigma_f\left[\frac{2}{N}\sum_{i=1}^N\left[\rho(\mathbf{z}_i;h)f(x_i)\right] \nabla_{\boldsymbol{\theta}}\left[\rho(\mathbf{z}_i;h)f(x_i)\right]\right].
\]
The above result is qutie similar to the paper; <font color="red">however, I am not sure why the paper can use two samples to calculate the mean $\left[\rho(\mathbf{z}_i;h)f(x_i)\right]$ and $\nabla_{\boldsymbol{\theta}}\left[\rho(\mathbf{z}_i;h)f(x_i)\right]$  separately to obtain an ubiased estimate of the gradient</font>.
* If we consider the function class from sieve-based functions up to degree $K$, we have the following $f$s (I am not familiar with the Gaussian Kernel idea in Algorithm 2 but we can construct $f$s as follows)
$$
\begin{aligned}
K=&1:\quad f_1(x_i)=w_1x_i^1\\
K=&2:\quad f_2(x_i)=w_1x_i^1+w_2x_i^2\\
K=&3:\quad f_3(x_i)=w_1x_i^1+w_2x_i^2+w_3x_i^3\\
\vdots\\
K=&K:\quad f_K(x_i)=w_1x_i^1+w_2x_i^2+w_3x_i^3+...+w_Kx_i^K,
\end{aligned}
$$
therefore, we can calculate the gradient as follows (given $\sigma_k$ and $h$)
\[
 \nabla_{\boldsymbol{\theta}}L(h,\boldsymbol{\sigma})=\sum_{k=1}^K\sigma_k\left[\frac{2}{N}\sum_{i=1}^N\left[\rho(\mathbf{z}_i;h)f_k(x_i)\right] \nabla_{\boldsymbol{\theta}}\left[\rho(\mathbf{z}_i;h)f_k(x_i)\right]\right],
\]
and we also need to calculate the gradient over $w$s for the functions $f$ to update $w$s,
\[
 \nabla_{\boldsymbol{w}}L(h,\boldsymbol{\sigma})=\sum_{k=1}^K\sigma_k\left[\frac{2}{N}\sum_{i=1}^N\left[\rho(\mathbf{z}_i;h)f_k(x_i)\right] \nabla_{\boldsymbol{w}}\left[\rho(\mathbf{z}_i;h)f_k(x_i)\right]\right].
\]