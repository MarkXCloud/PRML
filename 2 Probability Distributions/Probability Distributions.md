# Probability Distributions

[toc]

***

## 2.1 Binary Variables

#### Binary Distribution

$$
\begin{align}
	x&\in \{ 0,1 \}\\
	p(x&=1|\mu)=\mu\tag{2.1}\\
	p(x&=0|\mu)=1-\mu\\
	0&\le\mu\le1
\end{align}
$$

#### Bernoulli Distribution

$$
\begin{align}
	\textrm{Bern}(x|\mu)&=\mu^x(1-\mu)^{1-x}\tag{2.2}\\
	\mathbb{E}[x]&=\mu\tag{2.3}\\
	\textrm{var}[x]&=\mu(1-\mu)\tag{2.4}
\end{align}
$$

Now suppose we have a data set $\mathcal{D}=\{x_1,_\cdots,x_N \}$, of observed values of $x$. We can construct the likelihood function, which is a function of $\mu$, on the assumption that the observations are drawn independently from $p(x|\mu)$, so that
$$
p(\mathcal{D}|\mu)=\prod_{n=1}^{N}p(x_n|\mu)=\prod_{n=1}^{N}\mu^{x_n}(1-\mu)^{1-x_n}\tag{2.5}
$$
The log likelihood function is given by
$$
\ln{p(\mathcal{D}|\mu)}=\sum_{n=1}^{N}\ln p(x_n|\mu)=\sum_{n=1}^{N}\{x_n\ln\mu+(1-x_n)\ln(1-\mu) \}\tag{2.6}
$$
The sum $\sum_nx_n$ provides an example of a sufficient statistic for the data under this distribution. We then derivate with respect to $\mu$
$$
\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_n\tag{2.7}
$$
which is also known as the *sample mean*. And we denote the number of observations of $x=1$ within this data set by $m$, wecan rewrite (2.7) in the form
$$
\mu_{ML}=\frac{m}{N}\tag{2.8}
$$

#### Binomial Distribution

> Distribution of the number $m$ of observations of $x = 1$ given that the data set has size $N$.  

$$
\begin{align}
	\textrm{Bin}(m|N,\mu)&=\binom{N}{m}\mu_m(1-\mu)^(N-m)\tag{2.9}\\
	\binom{N}{m}&\equiv\frac{N!}{(N-m)!m!}\tag{2.10}\\
	\mathbb{E}[m]&\equiv\sum_{m=0}^{N}m\textrm{Bin}(m|N,\mu)\ =\ N\mu\tag{2.11}\\
	\textrm{var}[m]&\equiv\sum_{m=0}^{N}(m-\mathbb{E}[m])^2\textrm{Bin}(m|N,\mu)\ =\ N\mu(1-\mu)\tag{2.12}
\end{align}
$$

### 2.1.1 The beta distribution

#### Conjugacy

> If we choose a prior to be proportional to powers of $\mu$ and $1-\mu$, then the posterior distribution, which is proportional to the product of the prior and the likelihood function, will have the same functional form as the prior. This property is called *conjugacy* and we will see several examples of it later in this chapter.  

#### Beta distribution

$$
\begin{align}
	\textrm{Beta}(\mu|a,b)&=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}\tag{2.13}\\
	\Gamma(x)\textrm{ is the gamma}&\textrm{ function defined by (1.141)}\\
	\int\textrm{Beta}(\mu|a,b)\textrm{d}\mu&=1\tag{2.14}\\
	\mathbb{E}[\mu]&=\frac{a}{a+b}\tag{2.15}\\
	\textrm{var}[\mu]&=\frac{ab}{(a+b)^2(a+b+1)}\tag{2.16}
\end{align}
$$

The parameters $a$ and $b$ are often called *hyperparameters* because they control the distribution of the parameter $µ$.  

The posterior distribution of $µ$ is now obtained by multiplying the beta prior (2.13) by the binomial likelihood function (2.9) and normalizing. 

==Why multiply the binomial likelihood?==
$$
p(\mu|m,l,a,b)\propto\mu^{m+a-1}(1-\mu)^{l+b-1}\tag{2.17}
$$

* $l=N-m$
* (2.17) has the same functional dependence on µ as the prior distribution, reflecting the conjugacy properties of the prior with respect to the likelihood function.  

Indeed, it is simply another beta distribution, and its normalization coefficient can therefore be obtained by comparison with (2.13) to give
$$
p(\mu|m,l,a,b)=\frac{\Gamma(m+a+l+b)}{\Gamma{(m+a)}\Gamma{(l+b)}}\mu^{m+a-1}(1-\mu)^{l+b-1}\tag{2.18}
$$

* We observe $m$ observations of $x=1$ and $l$ observations of $x=0$ has been increase the value of $a$ by $m$ and $b$ by $l$, in going from the prior distribution to the posterior distribution.
* $a$ and $b$ in the prior are *effective number of observations* of $x=1$ and $x=0$.

**Furthermore, the posterior distribution can act as the prior if we subsequently observe additional data.**

1. Taking observations one at a time
2. After each observation updating the current posterior distribution by multiplying by the likelihood function for the new observation and then normalizing to obtain the new, revised posterior distribution.
3. At each stage, the posterior is a beta distribution with some total number of (prior and actual) observed values for $x = 1$ and $x = 0$ given by the parameters $a$ and $b$.  
4. Incorporation of an additional observation of $x = 1$ simply corresponds to incrementing the value of $a$ by $1$, whereas for an observation of $x = 0$ we increment $b$ by $1$.  

![image-20211118195608320](../pic/image-20211118195608320.png)

The evaluation of the predictive distribution of $x$ given $\mathcal{D}$
$$
p(x=1|\mathcal{D})=\int_{0}^{1}p(x=1|\mu)p(\mu|\mathcal{D})\textrm{d}\mu=\int_{0}^{1}\mu p(\mu|\mathcal{D})\textrm{d}\mu=\mathbb{E}[\mu|\mathcal{D}]\tag{2.19}
$$
Using the (2.18) for the posterior distribution $p(\mu|\mathcal{D})$, together with (2.15) for the mean of the beta distribution, we obtain
$$
p(x=1|\mathcal{D})=\frac{m+a}{m+a+l+b}\tag{2.20}
$$

> The total fraction of observations (both real observations and fictitious prior observations) that correspond to $x = 1$ 

Whether as we observe more and more data, the uncertainty represented by the posterior distribution will steadily decrease?

Consider a general Bayesian inference for parameter $\theta$ for which we observed a data set $\mathcal{D}$, described by the joint distribution $p(\theta,\mathcal{D})$.The following result
$$
\mathbb{E}_\boldsymbol{\theta}[\boldsymbol{\theta}]=\mathbb{E}_\mathcal{D}[\mathbb{E}_\boldsymbol{\boldsymbol{\theta}}[\boldsymbol{\theta}|\mathcal{D}]]\tag{2.21}
$$
where
$$
\begin{align}
	\mathbb{E}_\boldsymbol{\theta}[\boldsymbol{\theta}]&\equiv \int p(\boldsymbol{\theta})\boldsymbol{\theta}\textrm{d}\boldsymbol{\theta}\tag{2.22}\\
	\mathbb{E}_\mathcal{D}[\mathbb{E}_\boldsymbol{\theta}[\boldsymbol{\theta}|\mathcal{D}]]&\equiv\int\{\int\boldsymbol{\theta}_p(\boldsymbol{\theta}|\mathcal{D})\textrm{d}\boldsymbol{\theta} \}p(\mathcal{D})\textrm{d}\mathcal{D}\tag{2.23}
\end{align}
$$
says that the posterior mean of $\boldsymbol{\theta}$, averaged over the distribution generating the data, is equal to the prior mean of $\boldsymbol{\theta}$. Similarly,
$$
\textrm{var}_\boldsymbol{\theta}[\boldsymbol{\theta}]=\mathbb{E}_\mathcal{D}[\textrm{var}_\boldsymbol{\theta}[\boldsymbol{\theta}|\mathcal{D}]]+\textrm{var}_\mathcal{D}[\mathbb{E}_\boldsymbol{\theta}[\boldsymbol{\theta}|\mathcal{D}]]\tag{2.24}
$$

* The term on the left-hand side of (2.24) is the prior variance of $\boldsymbol{\theta}$
* On the right-hand side, the first term is the average posterior variance of $\boldsymbol{\theta}$, and the second term measures the variance in the posterior mean of $\boldsymbol{\theta}$.
* Note, however, that this result only holds on average.

***

## 2.2 Multinomial Variables

**We consider:**

$1$-of-$K$ scheme in which the variable is represented by a $K$-dimensional vector $\textbf{x}$ ==in which one of the elements $x_k$ equals $1$, and all remaining elements equal $0$.==

Take $K=6$ and $x_3=1$ as an example:
$$
\textbf{x}=(0,0,1,0,0,0)^\textrm{T}\tag{2.25}
$$
We denote the probability of $x_k=1$ by the parameter $\mu_k$, then the distribution of $\textbf{x}$:
$$
p(\textbf{x}|\boldsymbol{\mu})=\prod_{k=1}^{K}\mu_k^{x_k}\tag{2.26}
$$
where $\boldsymbol{\mu}=(\mu_1,_\cdots,\mu_K)^\textrm{T}$ satisfy $\mu_k\ge0$ and $\sum_k\mu_k=1$
$$
\sum_\textbf{x}p(\textbf{x}|\boldsymbol{\mu})=\sum_{k=1}^{K}\mu_k=1\tag{2.27}
$$
and that
$$
\mathbb{E}[\textbf{x}|\boldsymbol{\mu}]=\sum_\textbf{x}p(\textbf{x}|\boldsymbol{\mu})\textbf{x}=\boldsymbol{\mu}\tag{2.28}
$$
The corresponding likelihood function takes the form  
$$
p(\mathcal{D}|\boldsymbol{\mu})=\prod_{n=1}^N\prod_{k=1}^K\mu_k^{x_{nk}}=\prod_{k=1}^K\mu_k^{(\sum_nx_{nk})}=\prod_{k=1}^K\mu_k^{m_k}\tag{2.29}
$$
The likelihood function depends on the N data points only through the $K$ quantities
$$
m_k=\sum_nx_{nk}\tag{2.30}
$$
which represent the number of observations of $x_k=1$. These are called the *sufficient statistics* for this distribution.

Solve $\boldsymbol{\mu}$ using Lagrange multiplier $\lambda$
$$
\sum_{k=1}^Km_k\ln\mu_k+\lambda(\sum_{k=1}^K\mu_k-1)\tag{2.31}
$$
Derivative with respect to $\mu_k$ to zero, we obtain
$$
\mu_k=-m_k/\lambda\tag{2.32}
$$
Solve the $lambda$ give $\lambda=-N$, thus
$$
\mu_k^{ML}=\frac{m_k}{N}\tag{2.33}
$$
which is the fraction of the $N$ observations for which $x_k = 1$.  

#### Multinomial distribution

Consider the joint distribution of the quantities $m_1,_\cdots,m_K$, conditioned on $\boldsymbol{\mu}$ on $N$ observations. From (2.29) this takes the form
$$
\textrm{Mult}(m_1,m_2,_\cdots,m_K|\boldsymbol{\mu},N)=\binom{N}{m_1m_2{_\cdots}m_K}\prod_{k=1}^{K}\mu_k^{m_k}\tag{2.34}
$$
which is known as *multinomial distribution*.



### 2.2.1 The Dirichlet distribution

By inspection of the form of the multinomial distribution, we see that the conjugate prior is given by  
$$
p(\boldsymbol{\mu}|\boldsymbol{\alpha})\propto\prod_{k=1}^K\mu_k^{\alpha_k-1}\tag{2.37}
$$

* $\boldsymbol{\alpha}$ denotes $(\alpha_1,_\cdots,\alpha_K)^{\textrm{T}}$

Note that, because of the summation constraint, the distribution over the space of the {µk} is confined to a *simplex* of dimensionality K - 1, as illustrated for K = 3 in Figure 2.4.  

![image-20211122172339243](../pic/image-20211122172339243.png)

#### Dirichlet distribution

The normalized form for this distribution is by  
$$
\textrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha})=\frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}\prod_{k=1}^K\mu_k^{\alpha_k-1}\tag{2.38}
$$
while
$$
\alpha_0=\sum_{k=1}^K\alpha_k\tag{2.39}
$$
![image-20211122172849527](../pic/image-20211122172849527.png)

Multiplying the prior (2.38) by the likelihood function (2.34), we obtain the posterior distribution for the parameters ${µ_k}$ in the form  
$$
p(\boldsymbol{\mu}|\mathcal{D},\boldsymbol{\alpha})\propto p(\mathcal{D}|\boldsymbol{\mu})p(\boldsymbol{\mu}|\boldsymbol{\alpha})\propto \prod_{k=1}^K\mu_k^{\alpha_k+m_k-1}\tag{2.40}
$$
We see that the posterior distribution again takes the form of a Dirichlet distribution, confirming that the Dirichlet is indeed a conjugate prior for the multinomial.   

Determine the normalization coefficient by comparison with (2.38) so that  
$$
\begin{align}
	p(\boldsymbol{\mu}|\mathcal{D},\boldsymbol{\alpha})&=\textrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha}+\textbf{m})\\
	&=\frac{\Gamma(\alpha_0+N)}{\Gamma(\alpha_1+m_1)\cdots\Gamma(\alpha_K+m_K)}\prod_{k=1}^K\mu_k^{\alpha_k+m_k-1}\tag{2.41}
\end{align}
$$

* $\textbf{m}=(m_1,_\cdots,m_K)^\textrm{T}$