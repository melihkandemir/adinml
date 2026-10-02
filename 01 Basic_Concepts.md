# What is machine learning?

> **Notational conventions used throughout these notes.**
>
> **Probability.** $P(\cdot)$ denotes a probability — of an event, or of a discrete random variable taking a value, so that $P(X=x)$ is the probability mass function of a discrete $X$. Lowercase $p(\cdot)$ is reserved for the probability **density** of a continuous random variable, which is not itself a probability. Where a joint object mixes the two (a continuous feature vector with a discrete label, say) we write $p$ for the joint and $P$ for its discrete factors. Subscripts name the distribution being integrated against, e.g. $P_{S \sim \mathcal{D}^m}(\cdot)$. Probabilities always take parentheses; expectations, variances and covariances take brackets: $\mathbb{E}[\cdot]$, $\mathrm{Var}[\cdot]$, $\mathrm{Cov}[\cdot,\cdot]$, with subscripts naming the variable averaged over ($\mathbb{E}_{S}$, $\mathbb{E}_{x\sim\mathcal{D}}$). $\mathds{1}(\cdot)$ is the indicator function and $\log$ always the **natural** logarithm.
>
> **Data.** $m$ is the size of a data set $S$, $d$ the dimension of the feature space, $C$ the number of classes, $[m] := \{1,\ldots,m\}$. Calligraphic letters denote spaces: $\mathcal{X}$ the feature space, $\mathcal{Y}$ the label space, $\mathcal{H}$ a hypothesis class, $\mathcal{D}$ a data distribution over $\mathcal{X}\times\mathcal{Y}$, and $\mathcal{D}_{\mathcal{X}}$ its marginal on $\mathcal{X}$.
>
> **Losses and risks.** The pointwise loss is always written $\ell(y, \widehat{y})$ — **true label first, prediction second**. $R(h)$ is the true risk and $\widehat{R}_S(h)$ the empirical risk on $S$: **a subscript on a risk always names the sample it is computed on** ($\widehat{R}_S$ on $S$, $\widehat{R}_{S'}$ on a second sample $S'$, $\widehat{R}_B$ on a mini-batch $B$), never a purpose (training vs. testing) — when a risk is computed on a training or test split, we say so in words. A calligraphic $\mathcal{L}$ denotes a training objective (an empirical risk plus a regularizer).
>
> **Recurring symbols.** $h$ is a hypothesis, $f$ the labeling function and $f^*$ the Bayes predictor, $A$ a learning algorithm, $w$ a weight vector, $\epsilon$ an accuracy and $\delta$ a failure probability, $\alpha$ a learning rate, $\lambda$ a regularization coefficient. From Chapter 5a onwards, $\sigma_i \in \{-1,+1\}$ are Rademacher signs; the activation function of a neural network is therefore written $g(\cdot)$, not $\sigma(\cdot)$, to keep the two apart. $\sigma$ without a subscript is a standard deviation (and, in Chapter 9 only, the bandwidth of an RBF kernel, which is its universal name there).

**Definition:** “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E”. [Mitchell, 1997](http://www.cs.cmu.edu/~tom/mlbook.html)<!-- cite: mitchell1997ml | book | author={Mitchell, Tom M.}; title={Machine Learning}; publisher={McGraw-Hill}; year={1997} -->

(Mitchell's $P$ for "performance measure" is the only place in these notes where the letter $P$ does not denote a probability; from the next section on we call it the **loss** $\ell$ and the **risk** $R$.)

**Purpose:** Designing algorithms to solve T with maximum P and minimum:

  1. time complexity (same as in any application)
  2. space complexity (same as in any application)
  3. sample complexity (new!)

## Formal definitions

### Experience (E)

Our computer program observes the environment by means of a **data set**, a set of tuples called **data points**: 

$$S := \{ (x_1, y_1), \ldots, (x_m, y_m) \}.$$ 

Each tuple comprises an **input** observation $x_i \in \mathcal{X}$, also called a **feature vector**, and a corresponding output observation $y_i \in \mathcal{Y}$. 

The set $\mathcal{X}$ is called the **feature space** and the set $\mathcal{Y}$ a **label space**. In some contexts, $S$ is called a **sample**. 

We assume that there exists a **labeling function** $f: \mathcal{X} \rightarrow \mathcal{Y}$ such that $y_i = f(x_i)$, which is **unknown** to us.

We further assume that each observation in the data set is an **independent and identically distributed (i.i.d.)** sample from an unknown data distribution $\mathcal{D}$. An i.i.d. data set is denoted as $S := \{(x_i, y_i) \overset{i.i.d.}{\sim} \mathcal{D} : i \in [m]\}$. 

Imagine playing backgammon. Each roll of the dice is a sample. These are independent samples if consecutive rolls do not influence each other. The samples are identically distributed if we use the same dice throughout the game. The i.i.d. assumption is at the heart of machine learning. The same process needs to be repeated for an agent to be able to learn it, just like us humans. 

### Task (T) 

Our task is to predict the outputs $y_i$ from input observations $x_i$ as accurately as possible. In mathematical terms, we look for a **hypothesis** $h: \mathcal{X} \rightarrow \mathcal{Y}$ that approximates the unknown labeling function, i.e. it should satisfy $f(x) \approx h(x), \forall x \in \mathcal{X}$. 
   * If $\mathcal{Y}$ is a set with discrete elements, then our task is called **classification**, because it boils down to assigning an input $x_i$ to an appropriate class. Imagine $x_i$ is a picture expressed as an array of pixel intensity values, $\mathcal{Y}$ is then the type of the object in the foreground.
   * If $\mathcal{Y}$ is a continuous set such as $\mathbb{R}$, then our task is called **regression**. Imagine $x_i$ is the properties of a house (size, age, coordinates), $\mathcal{Y}$ can be its price.

We do not search over all conceivable maps $\mathcal{X} \rightarrow \mathcal{Y}$. We commit in advance to a **hypothesis class** $\mathcal{H} \subseteq \mathcal{Y}^{\mathcal{X}}$ and search inside it. This commitment is the **inductive bias** of the learner. Chapter 5 shows that learning is impossible without one.

### Performance Measure (P) 

We define what we mean by an **accurate prediction** via a **loss function**: $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}^+$ which maps a pair $(y,y')$ comprising a prediction $y'$ and a corresponding true label $y$ to a score inversely proportional to the quality of the prediction. Commonsense design choices for supervised learning problems are:

1. the **zero-one loss** defined as $\ell(y,y') := \mathds{1}(y \neq y')$ for classification, and
2. the **squared error** defined as $\ell(y,y') := (y-y')^2$ for regression.

Here $\mathds{1}$ is the indicator function that returns $1$ if the predicate in its argument holds (e.g. if $y$ differs from $y'$) and $0$ otherwise. Put together, we are interested in a **learning algorithm** $A(\cdot)$ that takes a data set $S$ as input and returns a hypothesis. Let us denote this hypothesis as $h_S$, where the subscript $S$ is to highlight its dependence on the data set. Then we can express the learning process as $h_S \gets A(S)$. We expect from this algorithm to minimize the **generalization error (true risk)**, which is defined for zero-one loss as:

$$R(h) := \mathbb{E}_{x \sim \mathcal{D}_{\mathcal{X}}}\big[ \ell(f(x), h(x)) \big] = P_{x \sim \mathcal{D}_{\mathcal{X}}}\big(f(x) \neq h(x)\big),$$

where the second equality is specific to the zero-one loss.

Some intuition before we dissect the notation. A new application of our program will keep receiving input observations after we ship it, and we care about how often it will be wrong on *those*, not on the ones we have already collected. Since future inputs arrive according to the same chances as the ones in $S$ (this is exactly what the i.i.d. assumption buys us), the fair way to score a hypothesis is: *average its loss over the chances of seeing each input*. That is all $R(h)$ is — a weighted average of the pointwise losses $\ell(f(x), h(x))$, one weight per input $x$, with inputs we are likely to see counted more heavily than rare ones. If the loss is the zero-one loss, this average counts each input where $h$ and $f$ disagree, weighted in the same way; an average of zeros and ones weighted by non-negative chances that sum to one is precisely a probability, which is the second equality above.

Two remarks unpack the expectation symbol. First, $\mathcal{D}_{\mathcal{X}}$, read "$\mathcal{D}$ marginalized on $\mathcal{X}$", is the distribution of input observations alone — the chances of encountering each $x$, with the label $y$ **integrated out**. In our current setup the label is a deterministic function of the input, $y = f(x)$, so knowing $x$ already pins down $y$; carrying the label along would tell us nothing new, and $R(h)$ averages only over $x$. (When we later encounter noisy data, labels will have a randomness of their own, and we will need the full joint distribution of the pair $(x,y)$ — a preview of that treatment appears in Chapter 5.) Second, you can read $\mathbb{E}_{x \sim \mathcal{D}_{\mathcal{X}}}$ as *"the long-run average of $\ell(f(x), h(x))$ over inputs $x$ drawn repeatedly from $\mathcal{D}_{\mathcal{X}}$"*. Probability theory will be covered properly in Chapter 4; for now, this long-run-frequency reading is all you need. In words, $R(h)$ is the true chance that our hypothesis will make a different prediction than the true labeling function on a freshly drawn input.

## The Empirical Risk Minimization (ERM) Paradigm

The goal of a learning algorithm $A$ is to find the hypothesis that minimizes the generalization error:

$$h_* := \arg \min_{h \in \mathcal{H}} R(h).$$

We cannot solve this optimization problem because we do not know $\mathcal{D}$ and $f$. We only have an idea about these unknowns via the data set $S$. Let us then use the data set to curate a quantity that approximates the generalization error:

$$\widehat{R}_S(h) := \frac{1}{m} \sum_{i \in [m]} \ell(y_i, h(x_i)) = \frac{|\{i \in [m] : h(x_i) \neq y_i\}|}{m}$$

where $[m] := \{1, \ldots, m\}$ and $|\cdot|$ counts the number of elements a set contains, e.g. $|\{1,2\}|=2$. The quantity $\widehat{R}_S(h)$ is called the **training error** or **empirical risk**. We can then find a suitable hypothesis by minimizing the empirical risk:

$$h_S := \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h).$$

This paradigm, called **Empirical Risk Minimization (ERM)**, is at the center of machine learning. Yet, it has certain weaknesses. For instance, consider the following solution:

$$
h_S(x) :=
\begin{cases}
y_i, & \text{if } \exists i \in [m] \text{ s.t. } x_i = x,\\
0, & \text{otherwise.}
\end{cases}
$$

This solution will provide us zero $\widehat{R}_S(h_S)$. It may look as if the problem is solved, but actually this is the moment where all problems start.

## Example: Polynomial curve fitting

We observe data like below for 50 input observations $x$ and the corresponding outputs $y$. We aim to find a function $y=f(x)$ that maps inputs to outputs.


```python
import matplotlib.pyplot as plt
import torch as th

th.manual_seed(2530)  # fix the random draws so that the figures are reproducible

# This is the true data generating process.
# We do not have access to this information in a typical machine learning problem
inputs = th.linspace(0, 1, 50, dtype=th.float64)
outputs = th.sin(2 * th.pi * inputs)
labels = outputs + th.randn(inputs.shape[0], dtype=th.float64) * 0.25

plt.plot(inputs, labels, 'bo')
plt.xlabel("inputs (x)")
plt.ylabel("labels (y)")
plt.show()
```


    
![Synthetic noisy observations of a sinusoidal labeling function.](fig/generated/01_Basic_Concepts_1.png)
    


Express our observations as a set of input-output tuples: $S = \{ (x_1, y_1), \cdots, (x_{50}, y_{50}) \}$. We would like to use them for two purposes: 
  * to **learn** the unknown function $f$,
  * to **evaluate** the success of the learning process (i.e. how well we expect the learned hypothesis to predict future observations).

Let us then split our data into two equal partitions randomly, one per purpose: $S = S_{train} \cup S_{test}$.

$S_{train}$ is called the **training set**. Our learning algorithm will use it to find a hypothesis that minimizes the empirical risk.

$S_{test}$ is called the **test set**. We will use it to check how well the empirical risk represents the true risk. In formal terms, our goal is to approximate $R(h_{S_{train}})$.

Both splits are data sets in their own right, so every quantity we have defined for $S$ applies to them verbatim. In particular, the empirical risk of a hypothesis $h$ computed on the training set is $\widehat{R}_{S_{train}}(h)$, and on the test set $\widehat{R}_{S_{test}}(h)$: the subscript names the sample the average is taken over, which is the whole content of the symbol. We will call these two quantities the **training error** and the **test error** of $h$. Since our hypothesis $h_{S_{train}}$ is chosen by minimizing $\widehat{R}_{S_{train}}$, the test error $\widehat{R}_{S_{test}}(h_{S_{train}})$ is the honest estimate of $R(h_{S_{train}})$: $S_{test}$ plays no role whatsoever in training, so $\widehat{R}_{S_{test}}(h_{S_{train}})$ behaves like a fresh measurement of how $h_{S_{train}}$ fares on new data.


```python
num_samples = inputs.shape[0]
num_train_samples = num_samples // 2 
# draw a random permutation of indices
idx = th.randperm(num_samples)
inputs_train = inputs[idx[:num_train_samples]]
labels_train = labels[idx[:num_train_samples]]
inputs_test = inputs[idx[num_train_samples:]]
labels_test = labels[idx[num_train_samples:]]
```

By visual inspection, we find that the input and the output have a nonlinear and periodic relationship. Let us then choose a hypothesis space suitable to this approximation, for instance a polynomial of degree $M$:

$$y(x) := \sum_{m=0}^M w_m x^m.$$

If $M=0$, then our model learns to fit a constant: $y(x) = w_0$. 

If $M=1$, then it fits a line: $y(x) = w_0 + w_1 x$. 

If $M=2$, it fits a parabola: $y(x)=w_0 + w_1 x + w_2 x^2$, and so on.


```python
class PolynomialModel:
    def __init__(self, num_polynomial_degrees=1, regularizer=0.0):
        self.M = num_polynomial_degrees
        self.regularization_factor = regularizer
      
    def extract_features(self, inputs):
        # Column m of the feature matrix Z holds x^m, for m = 0, ..., M
        powers = th.arange(self.M + 1, dtype=inputs.dtype)
        return inputs.unsqueeze(1) ** powers

    def learn(self, inputs, labels):
        Z = self.extract_features(inputs)
        ridge = self.regularization_factor * th.eye(Z.shape[1], dtype=Z.dtype)
        A = Z.T @ Z + ridge
        # w_S = (Z^T Z + lambda I)^{-1} Z^T y, obtained by solving the linear
        # system A w = Z^T y rather than by forming A^{-1} explicitly
        self.weights = th.linalg.solve(A, Z.T @ labels)

    def predict(self, inputs):
        return self.extract_features(inputs) @ self.weights
```

The **extract_features** function converts each one-dimensional observation to an $M-$ dimensional **feature vector**. This operation is called **feature extraction**. 

The **learn** function takes a set of inputs and their corresponding labels in its arguments and fits the unknown model parameters $w_0, \cdots, w_M$ to data. We will cover later how exactly it does that.

The **predict** function takes only inputs in its arguments. It predicts their corresponding labels itself, hence the name. The source code of advanced machine learning models we will see later on follows the same template.


```python
model = PolynomialModel(num_polynomial_degrees=3) # Create the model instance

# Training
model.learn(inputs_train, labels_train) 

# Testing
predictions_test = model.predict(inputs_test) 
root_mean_squared_error = ((labels_test - predictions_test) ** 2).mean().sqrt()

print(root_mean_squared_error.item())
```

    0.25155540073010746


Let us visualize all we have. In the plot below, the training set is shown as blue circles, the test set as green circles, and the predictions of our machine learning model as red circles. The black curve is the labeling function we are trying to find out. Note that we can observe only noisy evaluations of it, just as in real life. Data collection processes are affected by unintended factors, which we call **noise**. 


```python
plt.plot(inputs,outputs,'k-', label="True function")
plt.plot(inputs_train,labels_train,'bo', label="Training data")
plt.plot(inputs_test, labels_test,'go', label="Test data")
plt.plot(inputs_test,predictions_test,'ro', label="Test predictions")
plt.legend(loc="upper right")
plt.show()
```


    
![The true labeling function together with the training set, the test set, and the model's predictions on the test set.](fig/generated/01_Basic_Concepts_2.png)
    


The properties of the model we have to choose before running the learning algorithm are called **hyperparameters**. For instance, the polynomial degree $M$ is a hyperparameter. Let us simply try all options from 1 to 10 and see how they affect the model's behavior.

Since we have a regression problem, we follow common sense and choose the squared error loss function. We take its square root to bring it to the same scale as the label space:

$$\mathrm{RMSE}_S(h) := \sqrt{\widehat{R}_S(h)} = \sqrt{\frac{1}{m} \sum_{i \in [m]} (h(x_i) - y_i)^2}.$$

This loss function is called the **Root Mean Squared Error (RMSE)**. As with any subscripted risk, the sample named in $\mathrm{RMSE}_S$ is the one the average is taken over — in the experiment below, $\mathrm{RMSE}_{S_{train}}$ and $\mathrm{RMSE}_{S_{test}}$.


```python
# The performance score
def root_mean_squared_error(labels_true, predictions):
    return ((labels_true - predictions) ** 2).mean().sqrt()

Mmax = 10
train_errors = th.zeros(Mmax, dtype=th.float64)
test_errors = th.zeros(Mmax, dtype=th.float64)

for M in range(Mmax):
    # Create the model instance
    model = PolynomialModel(num_polynomial_degrees=M)

    # Training: fit parameters to data on the training split.
    model.learn(inputs_train, labels_train)
    predictions_train = model.predict(inputs_train)
    train_err = root_mean_squared_error(labels_train, predictions_train)

    # Testing: predict on the test split and evaluate performance.
    predictions_test = model.predict(inputs_test)
    test_err = root_mean_squared_error(labels_test, predictions_test)

    train_errors[M] = train_err
    test_errors[M] = test_err

plt.plot(th.arange(Mmax), train_errors, "bo-", label="Training")
plt.plot(th.arange(Mmax), test_errors, "ro-", label="Test")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.legend(loc="upper right")
plt.show()
```


    
![Training and test error as a function of the polynomial degree $M$.](fig/generated/01_Basic_Concepts_3.png)
    


**Expected:** As $M$ increases, the training error $\widehat{R}_{S_{train}}(h_{S_{train}})$ monotonically decreases.

**Surprise:** However, the test error $\widehat{R}_{S_{test}}(h_{S_{train}})$ decreases up to $M=5$, and then increases — slowly at first, then sharply: by $M=9$ it is almost five times its minimum. 

What may have gone wrong? Let us take a closer look into individual cases.


```python
model0 = PolynomialModel(num_polynomial_degrees=0) 
model0.learn(inputs_train, labels_train) 
pred_0 = model0.predict(inputs)

model1 = PolynomialModel(num_polynomial_degrees=1) 
model1.learn(inputs_train, labels_train) 
pred_1 = model1.predict(inputs)

model4 = PolynomialModel(num_polynomial_degrees=4) 
model4.learn(inputs_train, labels_train) 
pred_4 = model4.predict(inputs)

model9 = PolynomialModel(num_polynomial_degrees=9) 
model9.learn(inputs_train, labels_train) 
pred_9 = model9.predict(inputs)

fig, axs = plt.subplots(2, 2)

for ax in axs.flat:
    ax.plot(inputs, outputs,'k-', label="True f")
    ax.plot(inputs_train, labels_train,'ro', label="Tr data")

axs[0, 0].plot(inputs, pred_0,'b-', label="M=0")
axs[0, 1].plot(inputs, pred_1,'g-', label="M=1")
axs[1, 0].plot(inputs, pred_4,'c-', label="M=4")
axs[1, 1].plot(inputs, pred_9,'m-', label="M=9")

for ax in axs.flat:
    ax.legend(loc="upper right")
    ax.label_outer()


```


    
![Polynomial fits of degree $M \in \{0,1,4,9\}$ against the true function and training data, illustrating underfitting and overfitting.](fig/generated/01_Basic_Concepts_4.png)
    


The model had poor performance because it could not express a periodic relationship with a constant $M=0$ or a line $M=1$. Poor model fit due to limited model capacity is called **underfitting**. An underfitted model performs poorly on both training and test data, i.e. it has high $\widehat{R}_{S_{train}}(h_{S_{train}})$ and high $\widehat{R}_{S_{test}}(h_{S_{train}})$.

What goes on for $M=9$ is interesting. A polynomial with such high degree is flexible enough to make sharp turns. The model learned to use this flexibility to pass through every training point, chasing even the noise in the observed labels and turning sharply wherever a training point demands it; but then it missed the trend and diverged from the true labeling function, swinging far from it in the gaps between training points. It is called **overfitting** when a model fits to training data extremely well, i.e. low $\widehat{R}_{S_{train}}(h_{S_{train}})$, but delivers poor performance on test data, i.e. high $\widehat{R}_{S_{test}}(h_{S_{train}})$. 

**Remark.** Designing highly-expressive models while keeping them immune to overfitting is one of the fundamental problems of machine learning.

---

To mitigate overfitting, we need to devise an algorithm that encourages minimum empirical risk and penalizes the increase of model capacity. One choice could be as follows:

$$h_{S_{train}} := \arg \min_{h \in \mathcal{H}} \widehat{R}_{S_{train}}(h) + \lambda \sum_{m=0}^M w_m^2.$$

Here the term $\lambda \sum_{m=0}^M w_m^2$ is called a **regularizer** and $\lambda$ a **regularization coefficient**. Let us see what happens then.


```python
for M in range(Mmax):
    # Create the model instance
    model = PolynomialModel(num_polynomial_degrees=M, regularizer=0.01)

    # Training: fit parameters to data on the training split.
    model.learn(inputs_train, labels_train)
    predictions_train = model.predict(inputs_train)
    train_err = root_mean_squared_error(labels_train, predictions_train)

    # Testing: predict on the test split and evaluate performance.
    predictions_test = model.predict(inputs_test)
    test_err = root_mean_squared_error(labels_test, predictions_test)

    train_errors[M] = train_err
    test_errors[M] = test_err

plt.plot(th.arange(Mmax), train_errors, "bo-", label="Training")
plt.plot(th.arange(Mmax), test_errors, "ro-", label="Test")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.legend(loc="upper right")
plt.show()
```


    
![Training and test error versus polynomial degree with $L_2$ regularization, showing reduced overfitting compared to the unregularized fit.](fig/generated/01_Basic_Concepts_5.png)
    


As seen, having a regularizer term significantly mitigates overfitting when the model capacity, i.e. the polynomial degree, increases: the test error now stays essentially flat beyond the optimal degree instead of exploding.
