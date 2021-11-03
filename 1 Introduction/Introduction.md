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

![image-20211103103006491](../pic/image-20211103102924785.png)

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

**Problem Intro**:

* Suppose we <u>observe</u> a *real-valued input variable x* and we wish to use this observation to <u>predict</u> the value of a *real-valued target variable t*.
* The data for this example is generated from the function $sin(2πx)$ with random noise included in the target values.



