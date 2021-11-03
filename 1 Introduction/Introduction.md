# Introduction

[toc]

## 1.History 

> The problem of searching for patterns in data is a fundamental one and has a long and
> successful history.

* Tycho Brahe in the 16th century allowed ohannes Kepler to discover the empirical laws of planetary motion.
* The discovery of regularities in atomic spectra played a key role in the development and verification of quantum physics in the early twentieth century.

==The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities to take actions such as classifying the data into different categories.==

## 2.Recognize handwritten digits

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
