# Linear Models for Regression

[toc]

## 3.1 Linear Basis Function Models

#### Linear Regression

The simplest linear model for regression is one that involves a linear combination of the input variables  
$$
y(\textbf{x},\textbf{w})=w_0+w_1x_1+_\cdots+w_Dx_D\tag{3.1}
$$
where $\textbf{x}=(x_1,_\cdots,x_D)^{\text{T}}$

Extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form  
$$
y(\textbf{x},\textbf{w})=w_0+\sum_{j=1}^{M-1}w_j\phi_j(\textbf{x})\tag{3.2}
$$
where $\phi_j(\textbf{x})$ are known as *basis functions*.

### 3.1.1 Maximum likelihood and least squares

**We have:**

target $t$, deterministic function $y(\textbf{x},\textbf{w})$ with additive noise so that
$$
t=y(\textbf{x},\textbf{w})+\epsilon\tag{3.7}
$$
where $\epsilon$ is a zero mean Gaussian random variable with precision (inverse variance) $\beta$. Thus
$$
p(t|\textbf{x},\textbf{w},\beta)=\mathcal{N}(t|y(\textbf{x},\textbf{w}),\beta^{-1})\tag{3.8}
$$
In the case of a Gaussian conditional distribution of the form (3.8), the conditional mean will be simply
$$
\mathbb{E}[t|\textbf{x}]=\int tp(t|\textbf{x})\text{d}t=y(\textbf{x},\textbf{w})\tag{3.9}
$$
We group the target variables {$t_n$} into a column vector that we denote by $\textsf{t}$. Making the assumption that these data points are drawn independently from the distribution (3.8), we obtain the following expression for the likelihood function
$$
p(\textsf{t}|\textbf{X},\textbf{w},\beta)=\prod_{n=1}^N\mathcal{N}(t_n|\textbf{w}^{\text{T}}\phi(\textbf{x}_n),\beta^{-1})\tag{3.10}
$$
Taking the logarithm
$$
\begin{align}
	\ln p(\textsf{t}|\textbf{w},\beta)&=\sum_{n=1}^N\ln\mathcal{N}(t_n|\textbf{w}^{\text{T}}\phi(\textbf{x}_n),\beta^{-1})\\
	&=\frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi)-\beta E_D(\textbf{w})
\end{align}\tag{3.11}
$$
where the sum-of-squares error function is defined by
$$
E_D(\textbf{w})=\frac{1}{2}\sum_{n=1}^N\{t_n-\textbf{w}^{\text{T}}\phi(\textbf{x}_n)\}^2\tag{3.12}
$$

1. Consider $\textbf{w}$, The gradient of the log likelihood function (3.11) takes the form

$$
\nabla\ln p(\textsf{t}|\textbf{w},\beta)=\sum_{n=1}^N\{t_n-\textbf{w}^{\text{T}}\phi(\textbf{x}_n)\}\phi(\textbf{x}_n)^{\text{T}}\tag{3.13}
$$

Solving for $\textbf{w}$ we obtain  
$$
\textbf{w}_{\text{ML}}=\left(\boldsymbol{\Phi}^{\text{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\text{T}}\textsf{t}\tag{3.14}
$$
which are known as the *normal equations* for the least squares problem.

Here $\boldsymbol{\Phi}$ is an $N\times M$ matrix, called the *design matrix*, whose elements are given by $\Phi_{nj}=\phi_j(\textbf{x}_n)$, so that
$$
\boldsymbol{\Phi}=
\begin{pmatrix}
	\phi_0(\textbf{x}_1)&\phi_1(\textbf{x}_1)&\cdots&\phi_{M-1}(\textbf{x}_1)\\
	\phi_0(\textbf{x}_2)&\phi_1(\textbf{x}_2)&\cdots&\phi_{M-1}(\textbf{x}_2)\\
	\vdots&\vdots&\ddots&\vdots\\
	\phi_0(\textbf{x}_N)&\phi_1(\textbf{x}_N)&\cdots&\phi_{M-1}(\textbf{x}_N)
\end{pmatrix}\tag{3.16}
$$
The quality
$$
\boldsymbol{\Phi}^\dagger\equiv\left(\boldsymbol{\Phi}^{\text{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\text{T}}\tag{3.17}
$$
is known as the *Moore-Penrose pseudo-inverse* of the matrix $\boldsymbol{\Phi}$. It can be regarded as a generalization of the notion of matrix inverse to nonsquare matrices.   

If we make the bias parameter explicit, then the final result of $w_0$ becomes
$$
w_0=\bar{t}-\sum_{j=1}^{M-1}w_j\overline{\phi_j}\tag{3.19}
$$
where we have defined
$$
\bar{t}=\frac{1}{N}\sum_{n=1}^Nt_n,\ \ \ \ \overline{\phi_j}=\frac{1}{N}\sum_{n=1}^N\phi_j(\textbf{x}_n)\tag{3.20}
$$
Thus the bias $w_0$ compensates for the difference between the averages (over the training set) of the target values and the weighted sum of the averages of the basis function values.  

2. Consider $\beta$, giving

$$
\frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^N\{t_n-\textbf{w}_{\text{ML}}^{\text{T}}\phi(\textbf{x}_n)\}^2\tag{3.21}
$$

and so we see that the inverse of the noise precision is given by the residual variance of the target values around the regression function.  

### 3.1.2 Geometry of least squares

![image-20211201194054321](../pic/image-20211201194054321.png)

### 3.1.3 Sequential learning

We can obtain a sequential learning algorithm by applying the technique of *stochastic gradient descent*, also known as *sequential gradient descent*, as follows.

If the error function comprises a sum over data points $E=\sum_nE_n$, then after presentation of pattern $n$, the stochastic gradient descent algorithm updates the parameter vector $\textbf{w}$ using
$$
\textbf{w}^{(\tau+1)}=\textbf{w}^{(\tau)}-\eta\nabla E_n\tag{3.22}
$$

* $\tau$ is the iteration number
* $\eta$ is the learning rate

For the sum-of-squares error function (3.12), this gives
$$
\textbf{w}^{(\tau+1)}=\textbf{w}^{(\tau)}+\eta(t_n-\textbf{w}^{(\tau)\text{T}}\phi_n)\phi_n\tag{3.23}
$$
where $\phi_n=\phi(\textbf{x}_n)$. This is known as *least-mean-squares* or the *LMS* algorithm.  

### 3.1.4 Regularized least squares

Error function with regularization form
$$
E_D(\textbf{w})+\lambda E_W(\textbf{w})\tag{3.24}
$$
and becomes
$$
\frac{1}{2}\sum_{n=1}^N\{t_n-\textbf{w}^{\text{T}}\phi(\textbf{x}_n)\}^2+\frac{\lambda}{2}\textbf{w}^{\text{T}}\textbf{w}\tag{3.27}
$$
also known as *weight decay* because in sequential learning algorithms, it encourages weight values to decay towards zero, unless supported by the data. In statistics, it provides an example of a *parameter shrinkage* method because it shrinks parameter values towards zero.

Setting the gradient of (3.27) with respect to $\textbf{w}$ to zero, and solving for $\textbf{w}$ as before, we obtain
$$
\textbf{w}=\left(\lambda\textbf{I}+\boldsymbol{\Phi}^{\text{T}}\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^{\text{T}}\tag{3.28}
$$
A more generalized form is
$$
\frac{1}{2}\sum_{n=1}^N\{t_n-\textbf{w}^{\text{T}}\phi(\textbf{x}_n)\}^2+\frac{\lambda}{2}\sum_{j=1}^M|w_j|^q\tag{3.29}
$$
![image-20211201195735377](../pic/image-20211201195735377.png)

#### **Lasso regression**

$q=1$ is known as the *lasso* in the statistic literature.

It has the property that if $λ$ is sufficiently large, some of the coefficients $w_j$ are driven to zero, leading to a *sparse* model in which the corresponding basis functions play no role.   

To see this, we first note that minimizing (3.29) is equivalent to minimizing the unregularized sum-of-squares error (3.12) subject to the constraint  
$$
\sum_{j=1}^M|w_j|^q\le\eta\tag{3.30}
$$
for an appropriate value of the parameter $\eta$.

![image-20211201200410232](../pic/image-20211201200410232.png)