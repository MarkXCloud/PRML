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