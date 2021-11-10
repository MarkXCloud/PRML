# Introduction

[toc]

## 1.0 Preface

### 1.0.1.History 

> The problem of searching for patterns in data is a fundamental one and has a long and
> successful history.

* Tycho Brahe in the 16th century allowed ohannes Kepler to discover the empirical laws of planetary motion.
* The discovery of regularities in atomic spectra played a key role in the development and verification of quantum physics in the early twentieth century.

==The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities to take actions such as classifying the data into different categories.==

***



### 1.0.2.Recognize handwritten digits

<img src="../pic/image-20211103102924785.png" alt="image-20211103103006491" style="zoom:80%;" />

> Each digit corresponds to a 28×28 pixel image and so can be represented by a vector **x** comprising 784 real numbers

<u>**Goal**: build a machine that will take such a vector x as input and that will produce the identity of the digit 0, . . . , 9 as the output.</u>

Difficulty: It's tackled using handcrafted rules or heuristics for distinguishing the digits based on the shapes of the strokes.

**Better method:**

==Adopting a machine learning approach== 

We want:

* <u>a function y(x)</u> which takes a new digit image x as input and that generates an output vector y, encoded in the same way as the target vectors

We prepare: 

* a large set of N digits $\{x_1, . . . , x_N\} $called a <u>*training set*</u> is used to tune the parameters of an adaptive model, known in advance.
* express the category of a digit using *<u>target vector t</u>*, which represents the identity of the corresponding digit

We get:

* y(x) can then determine the identity of <u>new digit images</u>, which are said to comprise a <u>*test set*</u>
* ==The ability to categorize correctly new examples that differ from those used for training is known as *generalization*.==
* Generalization is a ==central goal== in pattern recognition, for the variability of the input vectors will be such that the training data can comprise only a tiny fraction of all possible input vectors.

In practice we do:

* the original input variables are typically *preprocessed* to transform them into some new space of variables where the pattern recognition problem will be easier to solve
* pre-processing stage is sometimes also called *feature extraction*. Note that new test data must be pre-processed using the same steps as the training data.
* the images  are typically translated and scaled into <u>a box of a fixed size</u>, reduce the variability
* pre-processing might also be performed in order to <u>speed up computation</u>
* care for information  *discarded*,  if this information is important to the solution of the problem then the overall accuracy of the system can suffer.

***



### 1.0.3.Domains of learning

1. ***supervised learning**:* training data comprises examples of the input vectors along with their corresponding target vectors. 
   * If the desired output consists of one or more continuous variables, then the task is called ***regression***.
2. ***unsupervised learning***: training data consists of a set of input vectors x without any corresponding target values. 
   * Discovering groups of similar examples within the data is called ***clustering***. 
   * Determining the distribution of data within the input space is known as ***density estimation***.
   * Project the data from a high-dimensional space down to two or three dimensions for the purpose of ***visualization***.
3. ***reinforcement learning***: finding suitable actions to take in a given situation in order to maximize a reward. Learning algorithm is not given examples of optimal outputs, but must instead discover them by a process of trial and error.
   * A general feature of reinforcement learning is the trade-off between *exploration*, in which the system tries out new kinds of actions to see how effective they are, and *exploitation*, in which the system makes use of actions that are known to yield a high reward. Too strong a focus on either exploration or exploitation will yield poor results.

***



## 1.1 Example: Polynomial Curve Fitting

### **Problem Intro**:

* Suppose we <u>observe</u> a *real-valued input variable x* and we wish to use this observation to <u>predict</u> the value of a *real-valued target variable t*.
* The data for this example is generated from the function $sin(2πx)$ with random noise included in the target values.

**What we have:**

A training set comprising N observations of x, written$ x ≡ (x_1, . . . , x_N)^T$, together with corresponding observations of the values of t, denoted $t ≡ (t_1, . . . , t_N)^T$, as shown below. The green curve shows the function $sin(2πx)$ used to generate
the data. Our goal is to predict the value of t for some new value of x, without knowledge of the green curve.

<img src="../pic/image-20211103135637589.png" alt="image-20211103135637589" style="zoom:80%;" />

> Why we add a noise?
>
> This noise might arise from ==intrinsically stochastic== (i.e. random) processes
> such as <u>radioactive decay</u> but more typically is due to <u>there being sources of variability that are themselves unobserved</u>.

**Our goal**:

exploit this training set in order to make predictions of the value $\hat{t}$ of the target variable for some new value $\hat{x}$ of the input variable.

**What we need**:

* *Probability theory* provides a framework for expressing such uncertainty in a precise and quantitative manner.
* *Decision theory* allows us to exploit this probabilistic representation in order to make predictions that are optimal according to appropriate criteria.

***



**We consider a simple approach**:

In particular, we shall fit the data using <u>a polynomial function of the form</u>:
$$
y(x,\textbf{w})=w_0+w_1x+w_2x^2+_\cdots+w_Mx^M=\sum_{j=0}^{M}w_jx^j
$$

* $\textit{M}$ is the *order* of the polynomial
* $x^j$ denotes $x$ raised to the power of $j$
* $w_0, _\cdots, w_M$  are collectively denoted by the vector $\textbf{w}$
* Our function is nonlinear to $x$, <u>but linear to  $\textbf{w}$</u>, so it's called *linear model*.



### **How we determine the coefficient $\textbf{w}$: minimize the error function**

By minimizing an *error function* that measures the misfit between the function $y(x,\textbf{w})$.

One simple choice of error function, which is widely used, is given by the sum of the squares of the errors between the predictions $y(x_n,\textbf{w})$ for each data point $x_n$ and the corresponding target values $t_n$, so that we minimize
$$
E(\textbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2
$$

> the factor 1/2 isincluded for later convenience

it is a nonnegative quantity that would be zero if, and only if, the function $y(x_n,\textbf{w})$ were to pass exactly through each training data point. As shown of the green bars below.

<img src="../pic/image-20211103142941240.png" alt="image-20211103142941240" style="zoom:80%;" />

* Since: the error function is <u>a quadratic function</u> of the coefficients $\textbf{w}$
* we can infer: its ==derivatives== with respect to the coefficients will be <u>linear</u> in the elements of $\textbf{w}$
* so we can get so the minimization of the error function has a unique solution,
  denoted by $\textbf{w}^\star$, which can be found in closed form.

***



### **How we choose the order $\textit{M}$ : Avoid overfitting**

This is an important concept called *model comparison* or *model selection*

<img src="../pic/image-20211103144018551.png" alt="image-20211103144018551" style="zoom:80%;" />

* $\textit{M}=0\ and \ 1$​ polynomials give rather poor fits to the data and consequently rather poor representations of the function $sin(2πx)$.
* $\textit{M}=3$​ polynomial seems to give the best fit to the function $sin(2πx)$ of the examples.
* $\textit{M}=9$ excellent fit. $E(\textbf{w}^\star)=0$​.However, the fitted curve oscillates wildly and gives a very poor representation of the function $sin(2πx)$. This latter behaviour is known as *over-fitting*.

**How we evaluate**:

Considering a separate test set comprising <u>100 data points</u> generated <u>using exactly the same procedure used to generate the training set points</u> but with <u>new choices for the random noise values</u> included in the target values.

We use root-mean-square (RMS) error defined by
$$
E_{RMS}=\sqrt{2E(\textbf{w}^\star)/N}
$$

* $\textit{N}$​ allows us to compare different sizes of data sets on an equal footing
* the square root ensures that $E_{RMS}$​ is measured on the same scale (and in the same units) as the target variable $t$.

Graphs of the root-mean-square error:

<img src="../pic/image-20211103150753399.png" alt="image-20211103150753399" style="zoom:80%;" />

> We only have 10 data points in the training set so when $\textit{M}=9$ it tuned exactly to them.

**We gain some insight**:

<img src="../pic/image-20211103151352930.png" alt="image-20211103151352930" style="zoom:80%;" />

$\textit{M}=9$ the coefficients have become finely tuned to the data <u>by developing large positive and negative values</u> so that the corresponding polynomial function matches each of the data points exactly, but between data points (particularly near the ends of the range) the function exhibits the large oscillations observed



**How model behaves as the size of the dataset is varied**:

For a given model complexity, ==the over-fitting problem become less severe as the size of the data set increases.==

<img src="../pic/image-20211103151730838.png" alt="image-20211103151730838" style="zoom:80%;" />

==Another way to say this is that the larger the data set, the more complex (in other words more flexible) the model that we can afford to fit to the data.==

>One rough heuristic that is sometimes advocated is that <u>the number of data points should be no less than some multiple (say 5 or 10) of the number of adaptive parameters</u> in the model.

***



### **How we limit the complexity and flexibility of the model: Regularization**

> One technique that is often used to control the over-fitting phenomenon in such cases is that of *regularization*.

Adding a <u>penalty term</u> to the error function in order to discourage the coefficients from reaching large values, leading to a modified error function of the form
$$
\tilde{E}(\textbf{w})=\frac{1}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2+\frac{\lambda}{2}||\textbf{w}||^2
$$

* $||\textbf{w}||^2\equiv \textbf{w}^T\textbf{w}=w_0^2+w_1^2+_\cdots+w_M^2$
* the coefficient $\lambda$​ governs the <u>relative importance of the regularization term compared with the sum-of-squares error term</u>



In particular, this quadratic regularizer is called *ridge regression* (Hoerl and Kennard, 1970). In the context of neural networks, this approach is known as *weight decay*.

![image-20211103154037395](../pic/image-20211103154037395.png)

<img src="../pic/image-20211103154117029.png" alt="image-20211103154117029" style="zoom:80%;" />

==The graph and the table show that regularization has the desired effect of reducing the magnitude of the coefficients.==

The impact of the regularization term on the generalization error can be seen by
plotting the value of the RMS error for both training and test sets against $ln λ$

<img src="../pic/image-20211103154410833.png" alt="image-20211103154410833" style="zoom:80%;" />

>We see that in effect $λ$ now controls the effective complexity of the model and hence determines the degree of over-fitting.

***

## 1.2 Probability Theory



==A key concept in the field of pattern recognition is that of uncertainty.==

**The Rules of Probability**
$$
\textbf{sum\ rule}\ \ p(X)=\sum_Yp(X,Y)\\
\textbf{product rule}\ \ p(X,Y)=p(Y|X)p(X)
$$
*Bayes' theorem*:
$$
p(Y|X)=\frac{p(X|Y)p(Y)}{p(X)}
$$
Using the sum rule, the denominator in Bayes’ theorem can be expressed in terms of the quantities appearing in the numerator
$$
p(X)=\sum_{Y}p(X|Y)p(Y)
$$
Finally, we note that if the joint distribution of two variables factorizes into the product of the marginals, so that
$$
p(X,Y)=p(X)p(Y)
$$
then $X$ and $Y$ are said to be *independent*.

***



### 1.2.1 Probability densities

The probability that $x$ will lie in an interval $(a,b)$ is then given by
$$
p(x\in (a,b))=\int_{a}^{b}p(x)dx
$$
Because probabilities are nonnegative, and because the value of $x$ must lie somewhere on the real axis, the probability density $p(x)$ must satisfy the two conditions
$$
p(x)\ge0\\
\int_{-\infty}^\infty p(x)dx=1
$$
If we consider a change of variables $x=g(y)$, the a function $f(x)$ becomes $\tilde{f}(y)=f(g(y))$. Note that $p_x(x)\delta x\simeq p_y(y)\delta y$. Then we have the transform below
$$
p_y(y)=p_x(x)|\frac{dx}{dy}|\\
=p_x(g(y))|g^\prime(y)|
$$
The probability that $x$ lies in the interval $(-\infty,z)$ is given by the *cumulative distribution function* defined by
$$
P(x)=\int_{-\infty}^{z}p(x)dx
$$
which satisfies $P^\prime (x)=p(x)$

<img src="../pic/image-20211103162701126.png" alt="image-20211103162701126" style="zoom:80%;" />

If we have several continues variables $x_1,_\cdots,x_D$, denoted collectively by the vector $\textbf{x}$, then we can define a joint probability density $p(\textbf{x})=p(x_1,_\cdots,x_D)$ such that the probability of $\textbf{x}$ falling in an interval volume $\delta\textbf{x}$ containing the point $\textbf{x}$ is given by $p(\textbf{x})\delta\textbf{x}$. This multivariate probability density must satisfy
$$
p(\textbf{x})\ge 0\\
\int p(\textbf{x})=1
$$
in which the integral is taken over the whole of $\textbf{x}$ space. We can also consider joint probability distributions over a combination of discrete and continuous variables.

The sum and product rules of probability, as well as Bayes’ theorem, apply equally to the case of probability densities, or to combinations of discrete and continuous
variables.
$$
p(x)=\int p(x,y)dy\\
p(x,y)=p(y|x)p(x)
$$

***



### 1.2.2 Expectations and covariances

One of the most important operations involving probabilities is that of finding <u>weighted averages of functions</u>.

a discrete distribution, it is given by
$$
\mathbb{E}[f]=\sum_x p(x)f(x)
$$
so that the average is weighted by the relative probabilities of the different values of $x$.

In the case of continuous variables, expectations are expressed in terms of an integration with respect to the corresponding probability density
$$
\mathbb{E}[f]=\int p(x)f(x)dx
$$
In either case, if we are given a finite number$N$ of points drawn from the probability distribution or probability density, then the expectation can be approximated as a finite sum over these points
$$
\mathbb{E}[f]\simeq \frac{1}{N}\sum_{n=1}^{N}f(x_n)
$$
The approximation becomes exact in the limit $N\rightarrow \infty$.

Sometimes we will be considering expectations of functions of several variables, in which case we can <u>use a subscript to indicate which variable is being averaged over</u>, so that for instance
$$
\mathbb{E}_x[f|y]=\sum_{x}p(x|y)f(x)
$$
denotes the average of the function $f(x,y)$ with respect to the distribution of $x$. Note that $\mathbb{E}_x[f(x,y)]$ will be a function of $y$.

We can also consider a *conditional expectation* with respect to a conditional distribution, so that
$$
\mathbb{E}_x[f|y]=\sum_xp(x|y)f(x)
$$
with an analogous definition for continuous variables.

The *variance* of $f(x)$ is defined by
$$
var[f]=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]
$$
and provides a measure of how much variability there is in $f(x)$ around its mean value $\mathbb{E}[f(x)]$.

Expanding out the square, we see that the variance can also be written in terms of the expectations of $f(x)$ and $f(x)^2$
$$
var[f]=\mathbb{E}[f(x)^2]-\mathbb{E}[f(x)]^2
$$
For two random variables $x$ and $y$, the *covariance* is defined by
$$
cov[x,y]=\mathbb{E}_{x,y}[\{x-\mathbb{E}[x]\}\{y-\mathbb{E}[y]\}]
\\
=\mathbb{E}_{x,y}[xy]-\mathbb{E}[x]\mathbb{E}[y]
$$
If x and y are independent, then their covariance vanishes.

In the case of two vectors of random variables $\textbf{x}$ and $\textbf{y}$, the covariance is a matrix
$$
cov[\textbf{x},\textbf{y}]=\mathbb{E}_{\textbf{x},\textbf{y}}[\{\textbf{x}-\mathbb{E}[\textbf{x}]\}\{\textbf{y}^T-\mathbb{E}[\textbf{y}^T]\}]\\
=\mathbb{E}_{\textbf{x},\textbf{y}}[{\textbf{x}\textbf{y}^T}]-\mathbb{E}[\textbf{x}]\mathbb{E}[\textbf{y}^T]
$$

***



### 1.2.3 Bayesian probabilities

> Now we turn to the more general *Bayesian* view, in which probabilities provide a quantification of uncertainty.

Some events that <u>cannot be repeated numerous times</u> in order to define a notion of probability . We would like to be able to quantify our expression of uncertainty and make precise revisions of uncertainty in the light of new evidence, as well as subsequently to be able to take optimal actions or decisions as a consequence.

>Cox (1946) showed that if numerical values are used to represent degrees of belief, then a simple set of axioms encoding common sense properties of such beliefs leads uniquely to a set of rules for manipulating degrees of belief that are equivalent to the sum and product rules of probability. This provided the first rigorous proof that probability theory could be regarded as an extension of Boolean logic to situations involving uncertainty (Jaynes, 2003).

In my word: probability theory can be regarded as an extention of Boolean logic in uncertainty situation.

Now, let us use the machinery of probability theory to describe the uncertainty in model parameters such as $\textbf{w}$ from a Bayesian perspective.

We capture our assumptions about $\textbf{w}$, before observing the data, in the form of a prior probability distribution $p(\textbf{w})$. The effect of the observed data $D = \{t_1, . . . , t_N\} $ is expressed through the conditional probability $p(D|\textbf{w})$

Bayes's theorem, which takes the form
$$
p(\textbf{w}|D)=\frac{p(D|\textbf{w})p(\textbf{w})}{p(D)}
$$
then allows us to evaluate the uncertainty in $\textbf{w}$ after we have observed $D$ in the form of the posterior probability $p(\textbf{w}|D)$.

The quantity $p(D|\textbf{w})$ on the right-hand side of Bayes’ theorem is evaluated for the observed data set D and can be viewed as a function of the parameter vector $\textbf{w}$, in which case it is called the *likelihood function*. It expresses how probable the observed data set is, for different settings of the parameter vector $\textbf{w}$.

Given this definition of likelihood, we can state Bayes’ theorem in words
$$
posterior \propto likelihood \times prior
$$
where all of these quantities are viewed as functions of $\textbf{w}$.

* In a frequentist setting, $\textbf{w}$ is considered to be a <u>fixed parameter</u>, whose value is determined by some form of ‘estimator’, and error bars on this estimate are obtained by considering the distribution of possible data sets $D$.
  * One approach to determining frequentist error bars is the *bootstrap*. Briefly, our original data set consists of $N$ data points $X=\{x_1,_\cdots,x_N\}$. We can create a new dataset $X_B$ by drawing $N$ points at random from $X$. This process can be repeated $L$ times to generate $L$ data sets each of size $N$ and each obtained by sampling from the original data set $X$​.The statistical accuracy of parameter estimates can then be evaluated by looking at the variability of predictions between the different bootstrap data sets.
* By contrast, from the Bayesian viewpoint there is only a single data set $D$ (namely the one that is actually observed), and the <u>uncertainty in the parameters is expressed through a probability distribution over $\textbf{w}$</u>.
  * One advantage of the Bayesian viewpoint is that the inclusion of prior knowledge arises naturally.

### 1.2.4 The Gaussian distribution

One of the most important probability distributions for continues variables: *normal* or *Gaussian* distribution.

For the case of a single real-valued variable $x$, the Gaussian distribution is defined
by
$$
N(x|\mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{\frac{1}{2}}}\exp{\{-\frac{1}{2\sigma^2}(x-\mu)^2\}}
$$

* $\mu$ is called the *mean*
* $\sigma^2$ is called the *variance*
* $\sigma$ is called the *standard deviation*
* $\beta=1/\sigma^2$ is called the *precision*

<img src="../pic/image-20211110105115553.png" alt="image-20211110105115553" style="zoom:80%;" />

We can see that the Gaussian distribution satisfied:
$$
N(x|\mu,\sigma^2)>0
$$
also:
$$
\int_{-\infty}^{\infty}N(x|\mu,\sigma^2)dx=1
$$
Thus the PDF satisfies the two requirements for a valid probability density.

We can readily find expectations of functions of x under the Gaussian distribution. In particular, the average value of $x$ is given by
$$
\mathbb{E}[x]=\int_{-\infty}^{\infty}N(x|\mu,\sigma^2)xdx=\mu
$$
Similarly, for the second order moment
$$
\mathbb{E}[x^2]=\int_{-\infty}^{\infty}N(x|\mu,\sigma^2)x^2dx=\mu^2+\sigma^2
$$
The variance of $x$ is given by
$$
var[x]=\mathbb{E}[x^2]-\mathbb{E}[x]^2=\sigma^2
$$
The maximum of a distribution is known as its **mode**. For a Gaussian, the mode coincides with the mean, for it is the highest point of the curve and have the most probability.

The Gaussian distribution defined over a $D$-dimensional vector $\textbf{x}$ of continuous variables is given by
$$
N(\textbf{x}|\mu,\Sigma)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}\exp{\{-\frac{1}{2}(\textbf{x}-\mu)^\textup{T}\Sigma^{-1}(\textbf{x}-\mu)\}}
$$

* The $D$-dimensional vector $\mu$ is  called the mean
* The $D\times D$ matrix $\Sigma$ is called the covariance
* The $|\Sigma|$ denotes the determinant of $\Sigma$.

***



Now suppose that we have a dataset of observations $\textbf{x}=(x_1,_\cdots,x_N)^T$,representing $N$ observations of scalar variable $x$. And our dataset  $\textbf{x}$ is i.i.d., we can write the probability of the dataset, given $\mu$ and $\sigma^2$, in the form
$$
p(\textbf{x}|\mu,\sigma^2)=\prod_{n=1}^{N}N(x_n|\mu,\sigma^2)
$$
When viewed as a function of $\mu$ and $\sigma^2$, this is the likelihood function for theGaussian and is interpreted diagrammatically in the Figure below

<img src="../pic/image-20211110112054953.png" alt="image-20211110112054953" style="zoom:80%;" />

The log likelihood function can be written in the form
$$
\ln{p}(\textbf{x}|\mu,\sigma^2)=-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2-\frac{N}{2}\ln\sigma^2-\frac{N}{2}\ln(2\pi)
$$
Maximizing with respect to $\mu$, we obtain the maximum likelihood solution given by
$$
\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_n
$$
which is the *sample mean*, i.e., the mean of the observed values $\{x_n\}$.

Similarly, maximizing with respect to $\sigma^2$, we obtain the maximum likelihood solution for the variance in the form
$$
\sigma_{ML}^2=\frac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{ML})^2
$$
which is the *sample variance* measured with respect to the sample mean $\mu_{ML}$.

An example of a phenomenon called *bias* and is related to the problem of over-fitting encountered in the text of polynomial curve fitting. We first note that the maximum likelihood solutions $μ_{ML}$ and $σ^2_{ML}$ are functions of the data set values $x_1, . . . , x_N$. Consider the expectations of these quantities with respect to the data set values, which themselves come from a Gaussian distribution with parameters $μ$ and $σ_2$. It is straightforward to show that
$$
\mathbb{E}[\mu_{ML}]=\mu\\
\mathbb{E}[σ^2_{ML}]=(\frac{N-1}{N})\sigma^2
$$
so that on average the maximum likelihood estimate will obtain the correct mean but will underestimate the true variance by a factor $(N-1)/N$. The intuition behind this result is given by Figure below

<img src="../pic/image-20211110150411418.png" alt="image-20211110150411418" style="zoom:80%;" />

The varience parameter unbiased:
$$
\tilde{\sigma}^2=\frac{N}{N-1}\sigma_{ML}^2=\frac{1}{N-1}\sum_{n=1}^{N}(x_n-\mu_{ML})^2
$$
Note that the bias of the maximum likelihood solution becomes less significant as the number N of data points increases, and in the limit $N → ∞$the maximum likelihood solution for the variance equals the true variance of the distribution that generated the data.

***



### 1.2.5 Curving fitting re-visited

Here we return to the curve fitting example and view it from a probabilistic perspective, thereby gaining some insights into error functions and regularization, as well as taking us towards a full Bayesian treatment.

**Our goal:**

* The goal in the curve fitting problem is to be able to make predictions for the target variable $t$ given some new value of the input variable $x$ on the basis of a set of training data comprising $N$ input values $x = (x_1, . . . , x_N)^T$ and their corresponding target values $t = (t_1, . . . , t_N)^T$.

**We make an assumption:**

Given the value of $x$, the corresponding value of $t$ has a Gaussian distribution with a mean equal to the value $y(x,\textbf{w})$ of the polynomial curve. Thus we have
$$
p(t|x,\textbf{w},\beta)=N(t|y(x,\textbf{w}),\beta^{-1})
$$

* We have defined a precision parameter $\beta$ corresponding to the inverse varience of the distribution.

<img src="../pic/image-20211110151753931.png" alt="image-20211110151753931" style="zoom:80%;" />

We now use the training data $\{\textbf{x,t}\}$ to determine the values of the unknown parameters $\textbf{w}$ and $\beta$ by maximum likelihood. If the data are assumed to be drawn independently from the distribution just above, then the likelihood functions is given by
$$
p(\textbf{t}|\textbf{x},\textbf{w},\beta)=\prod_{n=1}^{N}N(t_n|y(x_n,\textbf{w}),\beta^{-1})
$$
As we did before, it is convenient to maximize the logarithm of the likelihood function. We obtain the log likelihood function in the form
$$
\ln{p(\textbf{t}|\textbf{x},\textbf{w},\beta)}=-\frac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2+\frac{N}{2}\ln{\beta}-\frac{N}{2}\ln{(2\pi)}
$$
**Initially:**

1. We consider the $\textbf{w}_{ML}$. Since the last two terms on the right-hand side do not depend on $\textbf{w}$,  we omit them. 
2. Also, scaling factor does not effects the maximum procedure, so we replace the coefficient $\beta/2$ with $1/2$.
3. Finally, we transfer the maximizing on log into minimizing the negative log, which leads to minimizing the *sum-of-squares-error*. Thus the sum-of-squares error function has arisen as a consequence of maximizing likelihood under the assumption of a Gaussian noise distribution.

We can also use maximum likelihood to determine the precision parameter $\beta$ of the Gaussian conditional distribution. Maximizing with respect to $\beta$ gives
$$
\frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2
$$
Having determined the parameters $\textbf{w}$ and $β$, we can now make predictions for new values of $x$. Because we now have a probabilistic model, these are expressed in terms of the predictive distribution that gives the probability distribution over t, rather than simply a point estimate, and is obtained by substituting the maximum likelihood parameters to give
$$
p(t|x,\textbf{w}_{ML},\beta_{ML})=N(t|y(x,\textbf{w}_{ML}),\beta_{ML}^{-1})
$$
Now let us take a step towards a more Bayesian approach and introduce a prior distribution over the polynomial coefficients $\textbf{w}$. For simplicity, let us consider a Gaussian distribution of the form
$$
p(\textbf{w}|\alpha)=N(\textbf{w}|\textbf{0},\alpha^{-1}\textbf{I})=(\frac{\alpha}{2\pi})^{(M+1)/2}\exp{\{-\frac{\alpha}{2}\textbf{w}^\textup{T}\textbf{w}\}}
$$

* $\alpha$ is the precision of the distribution.
* $M+1$ is the total number of the elements in the vector $\textbf{w}$ for an $M^{th}$ order polynomial.
* Variables such as $\alpha$, which control the distribution of the model parameters, are called *hyperparameters*.

Using Bayes' theorem, the posterior distribution for $\textbf{w}$ is proportional to the product of the prior distribution and the likelihood function
$$
p(\textup{w}|\textbf{x,t},\alpha,\beta)\propto p(\textbf{t|x},\textup{w},\beta)p(\textup{w}|\alpha)
$$
we can now determine the $\textup{w}$ by finding the most probable value of $\textup{w}$ given the data, in other words by maximizing the posterior distribution. This technique is called *maximum posterior*, or simply *MAP*. Taking the negative logarithm of the posterior and combine with the expression of prior and likelihood, we find that the maximum of the posterior is given by the minimum of
$$
\frac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\textbf{w})-t_n\}^2+\frac{\alpha}{2}\textbf{w}^\textup{T}\textbf{w}
$$


Thus we see that maximizing the posterior distribution is equivalent to minimizing the regularized sum-of-squares error function encountered earlier in the form of polynomial episode, with a regularization parameter given by $\lambda=\alpha/\beta$.

