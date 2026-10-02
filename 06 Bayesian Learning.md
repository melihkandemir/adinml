# Maximum Likelihood Estimation

Machine learning is the tool of choice especially in real-world problems where there are factors of uncertainty and randomness. Having learned the principled way of accounting for uncertainty via probability theory, next we will see how we can use this theory to build machine learning models. Every machine learning problem has two key components: the data and the model. Let our data set be $S=\{x_1, x_2, \ldots, x_m\}$, where $x_i$ are the observed data points. At the stage of modeling, we draw a hypothesis about how these data could have been generated. Let our hypothesis be that the data are generated as independent samples from a distribution following a parametric probability function $P(x \mid \theta)$, where $\theta$ is the parameter of the distribution. Then the joint probability of the random variables representing the occurrence of the data set is given by

$$P(S \mid \theta) = P(x_1, x_2, \ldots, x_m \mid \theta) = \prod_{i=1}^m P(x_i \mid \theta).$$

This quantity, viewed as a function of $\theta$ for the fixed, observed $S$, is called the **likelihood** of $\theta$ given the data set $S$. The second equality follows from the assumption that the data points are independent. The maximum likelihood estimation (MLE) is a method of estimating the parameters $\theta$ by maximizing the likelihood $P(S \mid \theta)$. This way we aim to find the parameters that are most likely to have generated the data set $S$, which leads to the optimization problem below:

$$\theta_{MLE} = \arg\max_\theta P(S \mid \theta) = \arg\max_\theta \prod_{i=1}^m P(x_i \mid \theta).$$

Formulating the objective as a product of terms is numerically brittle: if each factor is small, the product of many such terms underflows the floating point system. It is also harder to differentiate a product than a sum. We therefore maximize the **log-likelihood** instead. Since the logarithm is monotonically increasing, its maximizer coincides with that of the likelihood itself:

$$\theta_{MLE} = \arg\max_\theta \sum_{i=1}^m \log P(x_i \mid \theta).$$

## Example: MLE for the Bernoulli distribution

Let $x \in \{0,1\}$ be the outcome of a coin toss ($1$ for heads, $0$ for tails), modeled as $P(x \mid \theta) := \theta^x (1-\theta)^{1-x}$, where $\theta \in [0,1]$ is the probability of heads. This is the **Bernoulli distribution**, and $\theta$ is its only parameter. Given $S = \{x_1,\ldots,x_m\}$, i.i.d. samples of $x$, the log-likelihood is

$$\log P(S \mid \theta) = \sum_{i=1}^m \log \theta^{x_i}(1-\theta)^{1-x_i} = \Big(\sum_{i=1}^m x_i\Big) \log \theta + \Big(m - \sum_{i=1}^m x_i\Big) \log(1-\theta).$$

Setting the derivative with respect to $\theta$ to zero,

$$
\begin{aligned}
\frac{\partial}{\partial \theta} \log P(S \mid \theta) &= \frac{\sum_{i=1}^m x_i}{\theta} - \frac{m - \sum_{i=1}^m x_i}{1-\theta} \triangleq 0\\
&\Rightarrow (1-\theta) \sum_{i=1}^m x_i = \theta\Big(m - \sum_{i=1}^m x_i\Big) \\
&\Rightarrow \theta_{MLE} = \frac{1}{m}\sum_{i=1}^m x_i.
\end{aligned}
$$

Unsurprisingly, the MLE of a Bernoulli parameter is simply the observed frequency of heads, i.e. the sample mean of $S$. This is the estimator we implicitly used throughout Chapter 4 whenever we wrote $\widehat\mu_m$ for a sample of Bernoulli random variables.

# Bayesian Learning

The MLE approach commits to a single value of $\theta$ and does not represent how uncertain we are about it — an uncertainty that is naturally large when $m$ is small (imagine estimating $\theta$ from three coin tosses) and shrinks as $m$ grows. This ansatz, where $\theta$ is treated as an unknown but fixed constant, is called **frequentist learning**. **Bayesian learning** instead treats $\theta$ itself as a random variable, with a distribution that captures our belief about its likely values before and after seeing data.

Concretely, the modeling assumption — the **generative process** or **data generating process** — becomes

$$
\begin{aligned}
 \theta &\sim p(\theta) \\
 x_i \mid \theta &\sim \mathrm{Bernoulli}(\theta), \quad i=1,\ldots,m.
\end{aligned}
$$

We start with a **prior distribution** $p(\theta)$ that encodes our belief about $\theta$ before seeing any data, and update it to a **posterior distribution** $p(\theta \mid S)$ after observing $S$, via Bayes' rule:

$$p(\theta \mid S) = \frac{P(S \mid \theta)\,p(\theta)}{P(S)} = \frac{P(S \mid \theta)\,p(\theta)}{\int_0^1 P(S \mid \theta')\,p(\theta') \, d\theta'}.$$

Note the case of the letters, which is not decoration: $P(S \mid \theta)$ and $P(S)$ are **probabilities** of the observed discrete data, while $p(\theta)$ and $p(\theta \mid S)$ are **densities** over the continuous parameter — the latter are not probabilities and can exceed $1$. Every mixed expression in this chapter obeys this rule, which is worth checking against as the formulas grow.

The denominator $P(S) = \int_0^1 P(S \mid \theta')p(\theta')\,d\theta'$ is the **evidence**. Two facts about it matter for the rest of this course:

* In almost every real-world model, the evidence is intractable to compute in closed form, forcing us to approximate the posterior. The whole field of Bayesian machine learning is, in large part, about finding good posterior approximations.
* The evidence measures how well the whole model family fits the data, averaged over the prior; it can therefore be used for model selection, by comparing $P(S)$ across candidate models.

The Bernoulli-parameter case is one of the rare instances that admits an exact, closed-form posterior. We choose the prior to be a **Beta distribution**, $\theta \sim \mathrm{Beta}(\alpha,\beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$ for hyperparameters $\alpha,\beta > 0$, because it is **conjugate** to the Bernoulli likelihood: the posterior comes out in the same family, which makes the update mechanical.

$$
\begin{aligned}
p(\theta \mid S) &\propto P(S \mid \theta)\, p(\theta) \\
&= \Big[\prod_{i=1}^m \theta^{x_i}(1-\theta)^{1-x_i}\Big] \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} \\
&= \theta^{\sum_{i=1}^m x_i + \alpha - 1} (1-\theta)^{m - \sum_{i=1}^m x_i + \beta - 1}.
\end{aligned}
$$

This is, up to a normalizing constant, exactly the density of a $\mathrm{Beta}(\alpha', \beta')$ distribution with

$$\begin{gathered}
\alpha' := \alpha + \sum_{i=1}^m x_i, \\
\beta' := \beta + m - \sum_{i=1}^m x_i.
\end{gathered}$$

Since a probability density integrates to one, the normalizing constant is forced to be the one that makes $\mathrm{Beta}(\alpha',\beta')$ a valid density, so $p(\theta \mid S) = \mathrm{Beta}(\theta \mid \alpha', \beta')$ exactly, with no approximation needed. Note the intuitive interpretation of the hyperparameters: $\alpha$ and $\beta$ act as **pseudo-counts** of heads and tails observed prior to seeing any real data, and the posterior simply adds the real counts $\sum_i x_i$ and $m - \sum_i x_i$ on top.

For a new observation $x_*$, the Bayesian model predicts a distribution over outcomes rather than a single value:

$$
\begin{aligned}
P(x_*=1 \mid S) &= \int_0^1 P(x_*=1 \mid \theta)\, p(\theta \mid S)\, d\theta = \int_0^1 \theta \cdot \mathrm{Beta}(\theta \mid \alpha', \beta')\, d\theta \\
&= \mathbb{E}_{\theta \sim p(\theta \mid S)}[\theta] = \frac{\alpha'}{\alpha'+\beta'},
\end{aligned}
$$

using the known mean of the Beta distribution. This is the **posterior predictive distribution**. Two of its properties are worth highlighting, both instances of a more general phenomenon in Bayesian models:

* It **averages over all possible values of $\theta$**, weighted by their posterior probability — a form of **model averaging** that accounts for parameter uncertainty, rather than committing to a single point estimate.
* As $m \to \infty$, the pseudo-counts $\alpha,\beta$ become negligible relative to $\sum_i x_i$ and $m - \sum_i x_i$, so $P(x_*=1 \mid S) \to \theta_{MLE}$: the Bayesian and frequentist predictions coincide in the large-sample limit, while differing — often substantially — for small $m$, exactly where accounting for uncertainty matters most.

Given a predictive distribution, there is more than one way to commit to a single prediction. The **Bayes predictor** takes the mode of the predictive distribution — its most probable value, $\arg\max_{x} P(x_*=x \mid S)$ — and is optimal in the sense of minimizing expected 0/1 loss. The **Gibbs predictor** instead samples a prediction, $\widehat{x} \sim P(x_* \mid S)$: it is correct exactly as often as the mode is, on average, but individual predictions vary.

# Maximum A-Posteriori Estimation

In most real-world applications, the posterior distribution is intractable because of the integral in the evidence term. One common approximation is the **maximum a-posteriori (MAP)** estimate, the mode of the posterior:

$$\theta_{MAP} = \arg\max_\theta p(\theta \mid S) = \arg\max_\theta P(S \mid \theta)p(\theta) = \arg\max_\theta \log P(S \mid \theta) + \log p(\theta).$$

The intractable evidence $P(S)$ drops out since it does not depend on $\theta$, making the objective tractable — this is the entire appeal of MAP over full Bayesian inference. For the Beta-Bernoulli model, since $\mathrm{Beta}(\alpha',\beta')$ has mode $\frac{\alpha'-1}{\alpha'+\beta'-2}$ (for $\alpha',\beta' > 1$),

$$\theta_{MAP} = \frac{\alpha + \sum_{i=1}^m x_i - 1}{\alpha + \beta + m - 2}.$$

Comparing this to $\theta_{MLE} = \frac{1}{m}\sum_i x_i$ makes the role of the prior explicit: MAP behaves like MLE run on an *augmented* data set that includes $\alpha - 1$ extra pseudo-heads and $\beta-1$ extra pseudo-tails. As $m \to \infty$, $\theta_{MAP} \to \theta_{MLE}$, the same way the full posterior predictive did above.

For a test input, the MAP estimate predicts as a **point estimate**, $x_* \sim P(x_* \mid \theta_{MAP})$, and does not perform model averaging: unlike the fully Bayesian predictive distribution above, it is not a truly Bayesian approach, even though it is derived from the posterior.

# Monte Carlo Integration

Bayesian quantities — evidences, predictive distributions, posterior expectations — are often integrals of the form

$$I = \int f(x)\, p(x)\, dx$$

for some function $f$ and probability density $p(x)$ (with the integral replaced by a sum against a probability mass function $P(x)$ in the discrete case), which is exactly what we needed to evaluate $\mathbb{E}_{\theta \sim p(\theta \mid S)}[\theta]$ above. When this integral has no closed form (as is typical outside conjugate models like Beta-Bernoulli), and we can draw $m$ samples $x_1,\ldots,x_m \sim p(x)$, we can approximate it by the sample average

$$I \approx \frac{1}{m}\sum_{i=1}^m f(x_i).$$

This is called **Monte Carlo integration**. By the weak law of large numbers (Chapter 4), this approximation is consistent: it converges to $I$ as $m \to \infty$.

# Generative Models

Consider a supervised learning problem with $(x,y) \sim \mathcal{D}$ for an unknown data distribution $\mathcal{D}$. Take the label $y$ to be discrete and the features $x$ to be continuous, so that — following the convention of Chapter 1 — the joint object is written $p(x,y)$ while its discrete factors are written with a capital $P$. There are two ways to factorize $p(x,y)$, and each suggests a different modeling strategy:

* $p(x,y) = P(y \mid x)\,p(x)$: model $P(y \mid x)$ directly and treat $p(x)$, the distribution of inputs alone, as something not worth modeling (it enters our formulas only as the distribution we average over — the same "average it out" treatment we gave the label in the marginal $\mathcal{D}_{\mathcal{X}}$ of Chapter 1; when it does appear, it can be approximated by Monte Carlo integration on the training inputs). This is called **discriminative modeling** — the logistic regression model of Chapter 3 is an example.
* $p(x,y) = p(x \mid y)\,P(y)$: model both $P(y)$ and $p(x \mid y)$, i.e. infer the whole data-generating process where a label is chosen first and the corresponding input is generated conditioned on it. This is called **generative modeling**.

## Example: a generative Bernoulli classifier

Let us build the simplest possible generative classifier: both the label $y \in \{0,1\}$ and a single binary feature $x \in \{0,1\}$ are Bernoulli. We choose

$$\begin{gathered}
y \sim \mathrm{Bernoulli}(\pi), \\
x \mid y \sim \mathrm{Bernoulli}(\theta_y),
\end{gathered}$$

so $\pi$ is the marginal probability of $y=1$, and $\theta_0, \theta_1$ are the class-conditional probabilities that $x=1$ given $y=0$ and $y=1$ respectively. Given $S = \{(x_i,y_i)\}_{i=1}^m$, all three parameters are Bernoulli parameters and each is fit by the MLE recipe derived above, applied to the relevant subset of the data:

$$\begin{gathered}
\widehat\pi = \frac{1}{m}\sum_{i=1}^m \mathds{1}(y_i{=}1), \\
\widehat\theta_c = \frac{\sum_{i: y_i=c} x_i}{\sum_{i:y_i=c} 1}, \quad c \in \{0,1\}.
\end{gathered}$$

The second formula is exactly $\theta_{MLE}$ from before, restricted to the subset of the data with $y_i=c$: fitting a generative model reduces to running the same single-distribution MLE machinery once per class. Given a new $x_*$, the model predicts

$$
\begin{aligned}
P(y{=}1 \mid x_*) &= \frac{P(x_*\mid y{=}1)\,P(y{=}1)}{P(x_*\mid y{=}0)P(y{=}0) + P(x_*\mid y{=}1)P(y{=}1)} \\
&= \frac{\widehat\theta_1^{x_*}(1-\widehat\theta_1)^{1-x_*}\,\widehat\pi}{\widehat\theta_0^{x_*}(1-\widehat\theta_0)^{1-x_*}(1-\widehat\pi) + \widehat\theta_1^{x_*}(1-\widehat\theta_1)^{1-x_*}\widehat\pi}.
\end{aligned}
$$

This is a complete, if minimal, generative classifier. Real problems have more than one feature, which brings us to Naive Bayes.

## Naive Bayes Classifier

Assume now $d$ discrete features $x = (x_1,\ldots,x_d)$, each taking values in a finite alphabet, and a label $y \in \{1,\ldots,C\}$. Modeling the *joint* class-conditional distribution $P(x \mid y{=}c)$ exactly would require a full contingency table with one entry per combination of feature values — exponential in $d$, and infeasible to estimate from any realistic amount of data. The **naive Bayes assumption** sidesteps this by assuming the features are conditionally independent given the class:

$$P(x \mid y{=}c) = \prod_{j=1}^d P(x_j \mid y{=}c).$$

This is "naive" because it is almost always false in the real world (features are typically correlated even within a class) — yet the resulting classifier is often highly effective in practice, and always tractable. For each feature $j$ with alphabet $\{1,\ldots,V_j\}$, we only need to estimate a small **tabular** distribution $P(x_j = v \mid y{=}c)$ per class $c$ and value $v$, i.e. a $C \times V_j$ table of probabilities. For binary features ($V_j=2$) this is again exactly the Bernoulli-MLE machinery from before, one Bernoulli parameter $\theta_{j,c}$ per (feature, class) pair:

$$\begin{gathered}
\widehat{\theta}_{j,c} := \widehat{P}(x_j{=}1 \mid y{=}c) = \frac{\sum_{i: y_i=c} x_{ij}}{m_c}, \\
m_c := \sum_{i: y_i=c} 1,
\end{gathered}$$

together with the class prior $\widehat\pi_c = m_c / m$. Given a query $x_*$, the classifier predicts

$$\widehat{y} = \arg\max_{c \in [C]} \widehat\pi_c \prod_{j=1}^d \widehat\theta_{j,c}^{\,x_{*j}} (1-\widehat\theta_{j,c})^{1-x_{*j}}.$$

The total number of estimated parameters is $O(Cd)$ — linear in the number of features, in stark contrast to the exponential table size of the unrestricted model. This tabular form generalizes to categorical (non-binary) features and to continuous features (e.g. via a per-class Gaussian $p(x_j \mid y{=}c) = \mathcal{N}(x_j \mid \mu_{j,c}, \sigma_{j,c}^2)$) by the same recipe, replacing the Bernoulli MLE with the MLE of the chosen per-feature family.

**PAC analysis of Naive Bayes.** Since every one of the $\widehat\theta_{j,c}$ (and $\widehat\pi_c$) is a Bernoulli-parameter MLE — a sample mean of a $\{0,1\}$-valued quantity — Hoeffding's inequality (Theorem 4.4) applies to each of them individually. We have $K = Cd$ such parameters $\widehat\theta_{j,c}$, each averaged over $m_c$ i.i.d. samples of a $[0,1]$-valued random variable, so the *uniform* Hoeffding bound of Theorem 4.5 applies directly with $K=Cd$ estimators and common range $b-a=1$:

$$P\Big(\exists (j,c) \in [d]\times[C] : |\widehat\theta_{j,c} - \theta_{j,c}| \geq \epsilon\Big) \leq 2\, C d \, e^{-2 m_{\min} \epsilon^2},$$

where $m_{\min} := \min_c m_c$ is the smallest class sample size, and the factor $2$ accounts for bounding the two-sided deviation $|\widehat\theta_{j,c}-\theta_{j,c}|\geq\epsilon$ by a union of the "too high" and "too low" one-sided events, each covered by Theorem 4.5. Setting the right-hand side to $\delta$ and solving for $\epsilon$, with probability at least $1-\delta$, **every** one of the $Cd$ estimated parameters is simultaneously within

$$\epsilon = \sqrt{\frac{\log(2Cd/\delta)}{2m_{\min}}}$$

of its true value. This is a genuine PAC-style guarantee: the sample complexity to achieve a target accuracy $\epsilon$ with confidence $1-\delta$ grows only as $O\big(\tfrac{1}{\epsilon^2}\log(Cd/\delta)\big)$ — logarithmically in the number of features $d$ and classes $C$, the direct benefit of the naive independence assumption reducing the effective number of parameters from exponential to linear in $d$. A uniform bound on the individual parameters like this one does not immediately give an equally tight bound on the resulting classifier's generalization error $R(\widehat{y})$ — the classification decision is a nonlinear function of all $Cd$ parameters at once — but it does show that the *building blocks* of the Naive Bayes classifier concentrate fast, which is the source of its practical robustness even from modest amounts of data.

# Implementation: Naive Bayes on a Real Data Set

Let us implement the tabular Naive Bayes classifier from scratch and test it on the **Wisconsin Breast Cancer** data set: $569$ real tumor samples, each with $30$ real-valued measurements (radius, texture, smoothness, ...) of a cell nucleus, labeled malignant or benign. Since our model above is built for *binary* features, we first binarize each measurement at its training-set median — a common, simple discretization strategy that brings a continuous data set into the tabular setting this chapter's math was derived for. Every other step — the train/test split, the class-conditional MLEs $\widehat\theta_{j,c}$, and the prediction rule — is implemented directly from the formulas above, with no black-box model classes.

```python
import torch as th
from sklearn.datasets import load_breast_cancer

# The only library-provided piece is the raw data set itself; the
# splitting, estimation, and prediction logic are all implemented below.
data = load_breast_cancer()
X_all = th.tensor(data.data, dtype=th.float64)
y_all = th.tensor(data.target, dtype=th.long)

th.manual_seed(0)
idx = th.randperm(y_all.numel())
n_train = int(0.7 * y_all.numel())
train_idx, test_idx = idx[:n_train], idx[n_train:]

# Binarize each feature at its training-set median.
median = X_all[train_idx].median(dim=0).values
X_train = (X_all[train_idx] > median).double()
X_test = (X_all[test_idx] > median).double()
y_train, y_test = y_all[train_idx], y_all[test_idx]

class BernoulliNaiveBayes:
    def __init__(self, num_classes=2, laplace=1.0):
        self.C = num_classes
        self.alpha = laplace  # Laplace (add-alpha) smoothing pseudo-count

    def learn(self, X, y):
        n, d = X.shape
        self.pi = th.zeros(self.C, dtype=th.float64)
        self.theta = th.zeros(self.C, d, dtype=th.float64)
        for c in range(self.C):
            Xc = X[y == c]
            n_c = Xc.shape[0]
            self.pi[c] = n_c / n
            # theta_{j,c} = (sum_i x_ij + alpha) / (n_c + 2*alpha): the MAP
            # estimate under a Beta(alpha+1, alpha+1) prior derived earlier
            # in this chapter, with alpha=1 recovering add-one smoothing.
            self.theta[c] = (Xc.sum(dim=0) + self.alpha) / (n_c + 2 * self.alpha)

    def predict(self, X):
        # Work in log-space for numerical stability, exactly as motivated
        # by the log-likelihood discussion at the start of this chapter.
        log_theta = th.log(self.theta)
        log_1m_theta = th.log(1 - self.theta)
        log_class_cond = X @ log_theta.T + (1 - X) @ log_1m_theta.T
        log_prior = th.log(self.pi)
        log_joint = log_class_cond + log_prior
        return th.argmax(log_joint, dim=1)

model = BernoulliNaiveBayes()
model.learn(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = (train_preds == y_train).double().mean().item()
test_acc = (test_preds == y_test).double().mean().item()

print(f"Naive Bayes train accuracy: {train_acc:.3f}")
print(f"Naive Bayes test accuracy:  {test_acc:.3f}")
```

    Naive Bayes train accuracy: 0.910
    Naive Bayes test accuracy:  0.918

Despite the crudeness of median binarization (which discards all information about *how far* a measurement is from the median) and the naive independence assumption across all $30$ features, the classifier reaches over $90\%$ test accuracy with a training procedure that is nothing more than counting — no gradient descent, no iterative optimization, just the closed-form MLE/MAP formulas of this chapter applied once per class.
