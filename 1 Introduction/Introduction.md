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
