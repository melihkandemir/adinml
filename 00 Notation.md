This table collects the symbols used throughout the book for quick reference. Chapter 1 states the underlying conventions (parentheses vs. brackets, uppercase vs. lowercase, etc.) in prose; this page is only a lookup table.

**Sets and spaces**

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | Feature (input) space |
| $\mathcal{Y}$ | Label (output) space |
| $\mathcal{H}$ | Hypothesis class |
| $\mathcal{D}$ | Data distribution over $\mathcal{X}\times\mathcal{Y}$ |
| $\mathcal{D}_{\mathcal{X}}$ | Marginal of $\mathcal{D}$ on $\mathcal{X}$ |
| $[m]$ | $\{1,\ldots,m\}$ |
| $\mathrm{dist}(\cdot,\cdot)$ | Distance function of a metric space (Chapter 2) |

**Data and samples**

| Symbol | Meaning |
|---|---|
| $S$ | A data set / sample |
| $m$ | Size of $S$ |
| $d$ | Dimension of the feature space |
| $C$ | Number of classes |
| $x, x_i$ | An input (feature vector) |
| $y, y_i$ | A true label |
| $\widehat{y}$ | A predicted label |
| $z, z_i$ | A score / pre-activation (also a training example in generic contexts) |

**Probability**

| Symbol | Meaning |
|---|---|
| $P(\cdot)$ | Probability of an event or of a discrete random variable's value |
| $p(\cdot)$ | Probability **density** of a continuous random variable |
| $\mathbb{E}[\cdot]$, $\mathrm{Var}[\cdot]$, $\mathrm{Cov}[\cdot,\cdot]$ | Expectation, variance, covariance |
| $\mathds{1}(\cdot)$ | Indicator function |
| $\log$ | Natural logarithm |
| $\sigma$ | Standard deviation (bandwidth of an RBF kernel in Chapter 9 only) |
| $\sigma_i \in \{-1,+1\}$ | A Rademacher sign (from Chapter 5a onwards) |

**Losses and risks**

| Symbol | Meaning |
|---|---|
| $\ell(y,\widehat{y})$ | Pointwise loss — true label first, prediction second |
| $R(h)$ | True risk (generalization error) of $h$ |
| $\widehat{R}_S(h)$ | Empirical risk of $h$ on $S$ — the subscript always names the sample (e.g. $\widehat{R}_{S_{train}}$ on the training split, $\widehat{R}_{S'}$ on a ghost sample, $\widehat{R}_B$ on a mini-batch) |
| $\mathcal{L}(\cdot)$ | Training objective (empirical risk plus a regularizer) |
| $R^*_{\mathcal{D}}$ | Bayes error of $\mathcal{D}$ |

**Learning-theoretic quantities**

| Symbol | Meaning |
|---|---|
| $h$ | A hypothesis |
| $h_S$ | A hypothesis learned from $S$ (the subscript names the sample the algorithm was run on) |
| $f$ | The true labeling function |
| $f^*$ | The Bayes predictor |
| $A$ | A learning algorithm |
| $\epsilon$ | An accuracy parameter |
| $\delta$ | A failure probability |
| $d_{VC}(\mathcal{H})$ | VC dimension of $\mathcal{H}$ |
| $\tau_{\mathcal{H}}(m)$ | Growth function of $\mathcal{H}$ |
| $\mathfrak{R}, \widehat{\mathfrak{R}}_S$ | Rademacher complexity (population, empirical) |

**Model and optimization symbols**

| Symbol | Meaning |
|---|---|
| $w$ | A weight vector |
| $\lambda$ | A regularization coefficient |
| $\alpha$ | A learning rate |

**Neural networks (Chapter 8)**

| Symbol | Meaning |
|---|---|
| $g(\cdot)$ | Activation function |
| $z_j^l$ | Pre-activation of neuron $j$ in layer $l$ |
| $h_j^l$ | Activation (output) of neuron $j$ in layer $l$ |
| $W_l, b^l$ | Weight matrix and bias vector of layer $l$ |
| $\delta_j^l$ | Backpropagated error of neuron $j$ in layer $l$ |

**Kernel methods (Chapter 9)**

| Symbol | Meaning |
|---|---|
| $k(\cdot,\cdot)$ | A kernel function |
| $\phi(\cdot)$ | A feature map |
| $\mathcal{H}_k$ | The RKHS of kernel $k$ |
| $K$ | A Gram matrix |
