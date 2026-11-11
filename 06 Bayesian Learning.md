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

Unsurprisingly, the MLE of a Bernoulli parameter is simply the observed frequency of heads, i.e. the sample mean of $S$. This is the estimator we implicitly used throughout Chapter 5 whenever we wrote $\widehat\mu_m$ for a sample of Bernoulli random variables.

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

Given a predictive distribution, there is more than one way to commit to a single prediction. The **Bayes predictor** takes the mode of the predictive distribution — its most probable value, $\arg\max_{x} P(x_*=x \mid S)$ — and is optimal in the sense of minimizing expected 0/1 loss. The **Gibbs predictor** instead samples a prediction, $\widehat{x} \sim P(x_* \mid S)$. It can never beat the mode, and the gap is bounded by a factor of two for every finite number of classes. To see this for $C$ classes, let $p = (p_1,\ldots,p_C)$ be the predictive distribution and $p_{\max} := \max_c p_c$. The Bayes error is $1-p_{\max}$, while the Gibbs error is

$$\sum_{c=1}^C p_c(1-p_c) = 1-\sum_{c=1}^C p_c^2 \;\leq\; 1-p_{\max}^2 = (1-p_{\max})(1+p_{\max}) \;\leq\; 2(1-p_{\max}),$$

since $p_{\max} \leq 1$. Thus the Gibbs predictor is never worse than twice the Bayes predictor, for every $C \geq 2$. The binary bound $2p(1-p) \leq 2\min(p,1-p)$ is the case $C=2$, in which the Gibbs error reduces to $2p(1-p)$, $p_{\max} = \max(p,1-p)$, and $1-p_{\max} = \min(p,1-p)$. For $C>2$ the bound is approached when one class has probability near one and the remaining mass is concentrated on a single alternative, so the factor two is not an artifact of the binary case.

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

This is called **Monte Carlo integration**. By the weak law of large numbers (Chapter 5), this approximation is consistent: it converges to $I$ as $m \to \infty$.

# Generative Models

Consider a supervised learning problem with $(x,y) \sim \mathcal{D}$ for an unknown data distribution $\mathcal{D}$. Take the label $y$ to be discrete and the features $x$ to be continuous, so that — following the convention of Chapter 2 — the joint object is written $p(x,y)$ while its discrete factors are written with a capital $P$. There are two ways to factorize $p(x,y)$, and each suggests a different modeling strategy:

* $p(x,y) = P(y \mid x)\,p(x)$: model $P(y \mid x)$ directly and treat $p(x)$, the distribution of inputs alone, as a quantity not worth modeling (it enters our formulas only as the distribution we average over — the same "average it out" treatment we gave the label in the marginal $\mathcal{D}_{\mathcal{X}}$ of Chapter 2; when it does appear, it can be approximated by Monte Carlo integration on the training inputs). This is called **discriminative modeling** — the logistic regression model of Chapter 4 is an example.
* $p(x,y) = p(x \mid y)\,P(y)$: model both $P(y)$ and $p(x \mid y)$, i.e. infer the whole data-generating process where a label is chosen first and the corresponding input is generated conditioned on it. This is called **generative modeling**.

## Example: a generative Bernoulli classifier

Let us build the simplest possible generative classifier: both the label $y \in \{0,1\}$ and a single binary feature $x \in \{0,1\}$ are Bernoulli. We choose

$$\begin{gathered}
y \sim \mathrm{Bernoulli}(\pi), \\
x \mid y \sim \mathrm{Bernoulli}(\theta_y),
\end{gathered}$$

so $\pi$ is the marginal probability of $y=1$, and $\theta_0, \theta_1$ are the class-conditional probabilities that $x=1$ given $y=0$ and $y=1$ respectively. Given $S = \{(x_i,y_i)\}_{i=1}^m$, all three parameters are Bernoulli parameters and each is fit by the MLE formula derived above, applied to the relevant subset of the data:

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

Assume now $d$ discrete features $x = (x_1,\ldots,x_d)$, with $x_j$ taking values in a finite alphabet of size $V_j$, and a label $y \in \{1,\ldots,C\}$. The previous subsection is the special case $d=1$, $C=2$, $V_1=2$, and the construction below extends it feature by feature. A generative classifier must model the class-conditional distribution $P(x \mid y{=}c)$. Its unrestricted form is a contingency table over the $\prod_{j=1}^d V_j$ feature configurations; for binary features this is $2^d$ entries per class, hence $(C-1)+C(2^d-1)$ free parameters, exponential in $d$. The MLE is the empirical frequency of each configuration within each class,

$$\widehat P(x=v \mid y{=}c) = \frac{N_{c,v}}{m_c}, \qquad N_{c,v} := \sum_{i:\,y_i=c} \mathds{1}(x_i=v), \qquad m_c := \sum_{i:\,y_i=c} 1,$$

so with $m_c$ samples of class $c$ at most $m_c$ configurations are observed, and every unobserved entry is estimated as zero. A query outside the observed set then receives likelihood zero under every class, so the classifier cannot rank the classes there. The obstruction is coverage, not arithmetic: if $x$ is uniform on $\{0,1\}^d$, a fixed configuration is unobserved with probability $(1-2^{-d})^{m_c}$, hence the expected fraction of unobserved configurations is at most $e^{-m_c/2^d}$, and making this fraction at most $\epsilon$ requires $m_c \geq 2^d \log(1/\epsilon)$ samples of the class. The unrestricted model therefore needs exponentially many samples before its table is even populated. A union-bound calculation over the table entries controls each entry to absolute error $\epsilon$ with $m_c=O(d/\epsilon^2)$ samples, but there are $2^d$ entries; summed over the table, the total-variation error can be as large as $2^d\epsilon$, so a constant-accuracy estimate of the joint distribution requires $\epsilon=O(2^{-d})$, hence $m_c=\Omega(2^d)$. The coverage argument makes the same obstruction explicit: the entries that no sample visits are exactly the entries on which the MLE commits to zero.

The **naive Bayes assumption** replaces the table by a product of $d$ one-dimensional tables,

$$P(x \mid y{=}c) = \prod_{j=1}^d P(x_j \mid y{=}c),$$

thereby sharing statistical strength across configurations: every sample, whatever its configuration, contributes to the estimates of all $d$ marginals. This is "naive" because it is almost always false in the real world (features are typically correlated even within a class) — yet the resulting classifier is often highly effective in practice, and always tractable. For each feature $j$ we estimate a **tabular** distribution $P(x_j = v \mid y{=}c)$ over $V_j$ values, i.e. $V_j-1$ free parameters per class, so the total number of free parameters is

$$(C-1) + C\sum_{j=1}^d (V_j-1) = O\Big(C\sum_{j=1}^d V_j\Big),$$

linear in $d$ for fixed alphabet sizes, in contrast to the exponential table of the unrestricted model. For binary features ($V_j=2$) this is exactly the Bernoulli-MLE machinery of the previous subsection, applied once per (feature, class) pair:

$$\begin{gathered}
\widehat{\theta}_{j,c} := \widehat{P}(x_j{=}1 \mid y{=}c) = \frac{\sum_{i: y_i=c} x_{ij}}{m_c}, \\
m_c := \sum_{i: y_i=c} 1,
\end{gathered}$$

together with the class prior $\widehat\pi_c = m_c / m$. Given a query $x_*$, the classifier predicts

$$\widehat{y} = \arg\max_{c \in [C]} \widehat\pi_c \prod_{j=1}^d \widehat\theta_{j,c}^{\,x_{*j}} (1-\widehat\theta_{j,c})^{1-x_{*j}}.$$

For binary features this is $Cd$ Bernoulli parameters plus $C-1$ prior parameters, linear in the number of features and in stark contrast to the exponential table size of the unrestricted model. This tabular form generalizes to categorical (non-binary) features and to continuous features (e.g. via a per-class Gaussian $p(x_j \mid y{=}c) = \mathcal{N}(x_j \mid \mu_{j,c}, \sigma_{j,c}^2)$) by the same procedure, replacing the Bernoulli MLE with the MLE of the chosen per-feature family.

**PAC analysis of Naive Bayes.** Since every one of the $\widehat\theta_{j,c}$ (and $\widehat\pi_c$) is a Bernoulli-parameter MLE — a sample mean of a $\{0,1\}$-valued quantity — Hoeffding's inequality (Theorem 5.6.5) applies to each of them individually. We have $K = Cd$ such parameters $\widehat\theta_{j,c}$. Conditionally on the labels $y_1,\ldots,y_m$, the estimate $\widehat\theta_{j,c}$ is an average of $m_c$ i.i.d. $\{0,1\}$-valued variables, so Hoeffding's inequality bounds each of its two tails by $e^{-2m_c\epsilon^2} \leq e^{-2m_{\min}\epsilon^2}$, and the union bound over the $2Cd$ one-sided events — the argument of the uniform Hoeffding bound, Theorem 5.6.6, with sample sizes $m_c \geq m_{\min}$ — gives

$$P\Big(\exists (j,c) \in [d]\times[C] : |\widehat\theta_{j,c} - \theta_{j,c}| \geq \epsilon\Big) \leq 2\, C d \, e^{-2 m_{\min} \epsilon^2},$$

where $m_{\min} := \min_c m_c$ is the smallest class sample size; the bound holds conditionally on the labels, hence also unconditionally whenever $m_{\min}$ is replaced by a deterministic lower bound on it. Setting the right-hand side to $\delta$ and solving for $\epsilon$, with probability at least $1-\delta$, **every** one of the $Cd$ estimated parameters is simultaneously within

$$\epsilon = \sqrt{\frac{\log(2Cd/\delta)}{2m_{\min}}}$$

of its true value. This is a genuine PAC-style guarantee: the sample complexity to achieve a target accuracy $\epsilon$ with confidence $1-\delta$ grows only as $O\big(\tfrac{1}{\epsilon^2}\log(Cd/\delta)\big)$ — logarithmically in the number of features $d$ and classes $C$, the direct benefit of the naive independence assumption reducing the effective number of parameters from exponential to linear in $d$. It is, however, a guarantee about *parameter estimation*, not about the risk of the classifier: the classification decision is a nonlinear function of all $Cd$ parameters at once, and errors in individual parameters can compensate one another in the argmax. The next section supplies the missing classifier-level analysis, replacing the $Cd$ parameters by the complexity of the set of decision rules they generate.

# Generalization Analysis of Naive Bayes

The Hoeffding argument of the previous subsection controls the $Cd$ estimated probabilities one at a time. That is a useful statement about *parameter estimation*, but it is not a risk bound for the classifier. The classifier is a nonlinear function of all the parameters at once: two parameter vectors can differ substantially while inducing the same decision rule, and small errors in many parameters can compensate one another. The object whose complexity governs generalization is therefore not the parameter vector but the set of decision rules that the parameters generate.

This section carries out the analysis in three steps. First, we show that the Naive Bayes classifier is a multiclass affine predictor, so its decision rules can be studied as a geometric hypothesis class rather than as a product of independent probability estimates. Second, we introduce the Natarajan dimension, the multiclass analogue of the VC dimension, and prove that the Naive Bayes class has Natarajan dimension at most $(C-1)(d+1)$. Third, we combine the multiclass Sauer--Shelah lemma with symmetrization and Massart's lemma to obtain a uniform convergence bound, from which we read off two guarantees: a risk bound for the Naive Bayes maximum-likelihood estimator, and an oracle inequality for empirical risk minimization over the same hypothesis class. Neither guarantee assumes that the data distribution satisfies the Naive Bayes factorization; the factorization is a property of the model, not an assumption about the world.

## Naive Bayes as a Multiclass Affine Predictor

The log-score of class $c$ in the Naive Bayes classifier is

$$\log\pi_c+\sum_{j=1}^d\Big[x_j\log\theta_{j,c}+(1-x_j)\log(1-\theta_{j,c})\Big].$$

Writing

$$w_{j,c}:=\log\frac{\theta_{j,c}}{1-\theta_{j,c}},\qquad b_c:=\log\pi_c+\sum_{j=1}^d\log(1-\theta_{j,c}),$$

turns it into the affine function $b_c+\langle w_c,x\rangle=\langle(b_c,w_c),(1,x)\rangle$. The next definition formalizes the resulting hypothesis classes.

**Definition 8.6.1 (Naive Bayes and affine hypothesis classes).** Let $\mathcal{X}=\{0,1\}^d$, $\mathcal{Y}=[C]$, and $\tilde{x}:=(1,x)\in\mathbb{R}^{d+1}$. For $v=(v_1,\ldots,v_C)\in\mathbb{R}^{C\times(d+1)}$, define $h_v:\mathcal{X}\to\mathcal{Y}$ by

$$h_v(x):=\arg\max_{c\in[C]}\ \langle v_c,\tilde{x}\rangle,$$

with ties broken in favor of the smallest class index. The class $\mathcal{H}_{\mathrm{aff}}:=\{h_v: v\in\mathbb{R}^{C\times(d+1)}\}$ is the class of **multiclass affine predictors**. The **Naive Bayes hypothesis class** $\mathcal{H}_{\mathrm{NB}}$ consists of all $h_\eta$ of the form

$$h_\eta(x):=\arg\max_{c\in[C]}\Big[\log\pi_c+\sum_{j=1}^d\big(x_j\log\theta_{j,c}+(1-x_j)\log(1-\theta_{j,c})\big)\Big],$$

where $\eta=(\pi,\theta)$ ranges over $\pi_c>0$, $\sum_{c=1}^C\pi_c=1$, and $\theta_{j,c}\in(0,1)$ for all $j\in[d]$ and $c\in[C]$, with the same tie-breaking rule.

---

The following proposition shows that the reparameterization is not merely a convenient rewriting: it exhausts the affine class.

**Proposition 8.6.2 (The Naive Bayes and affine classes coincide).** $\mathcal{H}_{\mathrm{NB}}=\mathcal{H}_{\mathrm{aff}}$.

***Proof.*** The inclusion $\mathcal{H}_{\mathrm{NB}}\subseteq\mathcal{H}_{\mathrm{aff}}$ follows from the reparameterization above, with $v_c=(b_c,w_c)$.

For the reverse inclusion, fix $v=(v_1,\ldots,v_C)\in\mathbb{R}^{C\times(d+1)}$ and write $v_c=(v_{c,0},v_{c,1},\ldots,v_{c,d})$. Set

$$\theta_{j,c}:=\frac{1}{1+e^{-v_{c,j}}}\in(0,1),\qquad \beta_c:=\sum_{j=1}^d\log(1-\theta_{j,c}),$$

and

$$\pi_c:=\frac{\exp(v_{c,0}-\beta_c)}{\sum_{c'=1}^C\exp(v_{c',0}-\beta_{c'})}.$$

Then $\pi_c>0$ and $\sum_c\pi_c=1$, so $\eta=(\pi,\theta)$ is a valid Naive Bayes parameter vector. Its score for class $c$ is

$$
\begin{aligned}
\log\pi_c+\sum_{j=1}^d\big(x_j\log\theta_{j,c}+(1-x_j)\log(1-\theta_{j,c})\big)
&=\log\pi_c+\beta_c+\sum_{j=1}^d x_j v_{c,j} \\
&=v_{c,0}+\sum_{j=1}^d x_j v_{c,j}-\log\Big(\sum_{c'=1}^C e^{v_{c',0}-\beta_{c'}}\Big) \\
&=\langle v_c,\tilde{x}\rangle-\mathrm{const},
\end{aligned}
$$

where the constant does not depend on $c$. Adding the same constant to every class score leaves the argmax and every tie unchanged, so $h_\eta=h_v$. Hence $\mathcal{H}_{\mathrm{aff}}\subseteq\mathcal{H}_{\mathrm{NB}}$, and the two classes are equal. $\square$

Proposition 8.6.2 is the formal content of the familiar observation that binary Naive Bayes is a linear classifier. It also exposes the difference between the number of parameters and the complexity of the decision rules. A Naive Bayes parameter vector has $C-1+Cd$ free coordinates, but the induced rule depends only on the differences between the class scores: adding the same vector to every $v_c$ leaves all comparisons unchanged. The space of score differences has dimension $(C-1)(d+1)$, which is the quantity measured by the next definition. Parameter counting therefore overstates the capacity of the classifier; what matters is the set of decision rules, not the coordinates used to describe it.

## Natarajan Dimension

The VC dimension of Chapter 6 is a binary notion: it asks whether a class can realize all $2^m$ labelings of an $m$-point set. For $C$ classes there are $C^m$ labelings, and requiring all of them is far too strong: even the class of all constant classifiers realizes only $C$ of them. Natarajan's definition replaces the Boolean cube $\{0,1\}^m$ by a Boolean subcube of the $C$-valued labelings. It chooses two witness labels at each point and requires the class to realize every one of the $2^m$ ways of switching between them.

**Definition 8.6.3 (Natarajan shattering and dimension).** Let $\mathcal{H}\subseteq\mathcal{Y}^{\mathcal{X}}$ with $|\mathcal{Y}|=C\ge2$. A set $S=\{x_1,\ldots,x_m\}\subseteq\mathcal{X}$ is **N-shattered** by $\mathcal{H}$ if there exist functions $f_1,f_2:S\to\mathcal{Y}$ with $f_1(x_i)\neq f_2(x_i)$ for every $i\in[m]$ such that for every subset $B\subseteq[m]$ there is $h\in\mathcal{H}$ satisfying

$$h(x_i)=f_1(x_i)\ \text{for } i\in B,\qquad h(x_i)=f_2(x_i)\ \text{for } i\notin B.$$

The **Natarajan dimension** $d_{\mathrm{N}}(\mathcal{H})$ is the largest cardinality of an N-shattered set, with $d_{\mathrm{N}}(\mathcal{H}):=\infty$ if arbitrarily large sets are N-shattered, and with the convention that the supremum over an empty family is $0$.

---

For $C=2$, taking $f_1\equiv1$ and $f_2\equiv0$ on $S$ recovers the usual definition of shattering, so $d_{\mathrm{N}}(\mathcal{H})=d_{VC}(\mathcal{H})$ for binary classes. The definition is due to [Natarajan (1989)](https://doi.org/10.1007/BF00116519)<!-- cite: natarajan1989learning | article | author={Natarajan, Balas K.}; title={On Learning Sets and Functions}; journal={Machine Learning}; year={1989}; volume={4}; number={1}; pages={67--97} -->. For $C>2$, N-shattering is weaker than requiring all $C^m$ labelings, but it is strong enough to force a polynomial growth function (Lemma 8.6.5 below). Unlike a parameter count, $d_{\mathrm{N}}$ is a property of the decision rules and is invariant under reparameterization.

**Theorem 8.6.4 (Natarajan dimension of affine predictors).** For $C\ge2$ and $d\ge1$,

$$d_{\mathrm{N}}(\mathcal{H}_{\mathrm{aff}})\le(C-1)(d+1).$$

***Proof.*** Let $S=\{x_1,\ldots,x_m\}\subseteq\mathcal{X}$ be N-shattered by $\mathcal{H}_{\mathrm{aff}}$, with witness functions $f_1,f_2:S\to[C]$. For $x\in S$ and $c\in[C]$, let $\Psi(x,c)\in\mathbb{R}^{C\times(d+1)}$ be the matrix whose $c$-th row is $\tilde{x}$ and whose other rows are zero; then $\langle v,\Psi(x,c)\rangle=\langle v_c,\tilde{x}\rangle$ is the score of class $c$ at $x$. Define

$$z_i:=\Psi(x_i,f_1(x_i))-\Psi(x_i,f_2(x_i)),\qquad i\in[m].$$

Since $f_1(x_i)\neq f_2(x_i)$, we have $z_i\neq0$. We claim that $z_1,\ldots,z_m$ are linearly independent.

Suppose otherwise, so that $\sum_{i=1}^m\alpha_i z_i=0$ for some coefficients $\alpha_i\in\mathbb{R}$, not all zero, and let $B:=\{i\in[m]:\alpha_i>0\}$. By N-shattering, there are hypotheses $v_B$ and $v_{B^c}$ realizing the two complementary labelings: $h_{v_B}(x_i)=f_1(x_i)$ for $i\in B$ and $h_{v_B}(x_i)=f_2(x_i)$ for $i\notin B$, and the opposite assignment for $v_{B^c}$. If $h_v(x_i)=f_1(x_i)$, then $f_1(x_i)$ is a maximizer of the scores at $x_i$, so $\langle v,z_i\rangle\ge0$; if $h_v(x_i)=f_2(x_i)$, then $\langle v,z_i\rangle\le0$. Hence

$$
0=\Big\langle v_B,\sum_{i=1}^m\alpha_i z_i\Big\rangle
=\sum_{i=1}^m\alpha_i\langle v_B,z_i\rangle\ge0,
$$

where each term is nonnegative because $\alpha_i>0$ on $B$ and $\alpha_i\le0$ outside $B$. Thus every term vanishes, and $\langle v_B,z_i\rangle=0$ whenever $\alpha_i\neq0$. Repeating the computation with $B$ replaced by $B^c$ (the terms are then nonpositive) gives $\langle v_{B^c},z_i\rangle=0$ for every $i$ with $\alpha_i\neq0$.

At such an $i$, both $v_B$ and $v_{B^c}$ assign equal scores to $f_1(x_i)$ and $f_2(x_i)$. Each hypothesis predicts one of these two labels at $x_i$, so both labels are tied at the maximum, and the fixed tie-breaking rule returns the same one of them for both hypotheses. But the labelings realized by $v_B$ and $v_{B^c}$ are complementary, so they prescribe opposite labels at every point of $S$, and in particular at every $i$ with $\alpha_i\neq0$. This is a contradiction. Hence $z_1,\ldots,z_m$ are linearly independent.

Finally, the sum of the rows of $z_i$ is $\tilde{x}_i-\tilde{x}_i=0$, so $z_i$ belongs to the subspace

$$\mathcal{U}:=\Big\{u\in\mathbb{R}^{C\times(d+1)}:\sum_{c=1}^C u_c=0\Big\}.$$

The map $u\mapsto\sum_{c=1}^C u_c$ is onto $\mathbb{R}^{d+1}$, so $\dim\mathcal{U}=C(d+1)-(d+1)=(C-1)(d+1)$. Linear independence gives $m\le\dim\mathcal{U}$. Since this holds for every N-shattered $S$, the claim follows. $\square$

Together with Proposition 8.6.2, Theorem 8.6.4 gives

$$d_{\mathrm{N}}(\mathcal{H}_{\mathrm{NB}})=d_{\mathrm{N}}(\mathcal{H}_{\mathrm{aff}})\le(C-1)(d+1).$$

This is the multiclass analogue of the VC dimension $d+1$ of affine binary classifiers. The bound is linear in the number of features and in the number of classes; the exponential dependence on $d$ that a full contingency table would suggest is absent. It is a bound on the *decision rules*, not on the $Cd+C-1$ parameters of the Naive Bayes parameterization.

## Multiclass Uniform Convergence

We now pass from combinatorial capacity to a risk bound. Recall from Definition 6.4.4 that the growth function of a hypothesis class $\mathcal{H}$ is

$$\tau_{\mathcal{H}}(m):=\max_{S\subseteq\mathcal{X},\,|S|=m}|\mathcal{H}_S|,\qquad \mathcal{H}_S:=\{(h(x_1),\ldots,h(x_m)):h\in\mathcal{H}\}.$$

It applies verbatim to $C$-valued classes, as nothing in that definition uses binary labels. The multiclass Sauer--Shelah lemma bounds $\tau_{\mathcal{H}}$ in terms of the Natarajan dimension.

**Lemma 8.6.5 (Multiclass Sauer--Shelah).** Let $\mathcal{H}\subseteq[C]^{\mathcal{X}}$ with $d_{\mathrm{N}}(\mathcal{H})=d_{\mathrm{N}}<\infty$. Then for every $m\ge1$,

$$\tau_{\mathcal{H}}(m)\le\sum_{i=0}^{\min\{d_{\mathrm{N}},m\}}\binom{m}{i}(C-1)^i.$$

In particular, for $m\ge d_{\mathrm{N}}$,

$$\tau_{\mathcal{H}}(m)\le\Big(\frac{e(C-1)m}{d_{\mathrm{N}}}\Big)^{d_{\mathrm{N}}}.$$

---

Lemma 8.6.5 is the multiclass counterpart of the Sauer--Shelah--Perles lemma (Lemma 6.4.8). The sum has the same shape, with the binary factor $1$ replaced by the $C-1$ alternative labels available each time the Natarajan dimension is charged; for $C=2$ it reduces exactly to Lemma 6.4.8. The result is due to [Natarajan (1989)](https://doi.org/10.1007/BF00116519)<!-- cite: natarajan1989learning | article | author={Natarajan, Balas K.}; title={On Learning Sets and Functions}; journal={Machine Learning}; year={1989}; volume={4}; number={1}; pages={67--97} -->; a complete proof appears in [Shalev-Shwartz & Ben-David (2014)](https://doi.org/10.1017/CBO9781107298019)<!-- cite: shalev2014understanding | book | author={Shalev-Shwartz, Shai and Ben-David, Shai}; title={Understanding Machine Learning: From Theory to Algorithms}; publisher={Cambridge University Press}; year={2014} -->, Chapter 29. The closed form follows from the sum by the same binomial estimate as in Lemma 6.4.8, using $m\ge d_{\mathrm{N}}$ and $C-1\ge1$.

**Theorem 8.6.6 (Multiclass uniform convergence).** Let $\mathcal{H}\subseteq[C]^{\mathcal{X}}$, let $\ell$ be the zero-one loss, and let $\tau_{\mathcal{H}}$ be the growth function. For every distribution $\mathcal{D}$ on $\mathcal{X}\times[C]$ and every $\delta\in(0,1)$, with probability at least $1-\delta$ over $S\sim\mathcal{D}^m$,

$$\sup_{h\in\mathcal{H}}\big|R(h)-\widehat{R}_S(h)\big|\le\sqrt{\frac{16\log\big(2\tau_{\mathcal{H}}(m)\big)+\log(1/\delta)}{m}}.$$

***Proof.*** Write $\Phi(S):=\sup_{h\in\mathcal{H}}|R(h)-\widehat{R}_S(h)|$.

*Step 1: concentration around the mean.* Changing one example of $S$ changes $\widehat{R}_S(h)$ by at most $1/m$ for every $h$ simultaneously, hence changes $\Phi(S)$ by at most $1/m$. Thus $\Phi$ has bounded differences $c_i=1/m$ and $\sum_{i=1}^m c_i^2=1/m$. McDiarmid's inequality (Theorem 7.1.1) gives, with probability at least $1-\delta$,

$$\Phi(S)\le\mathbb{E}_S[\Phi(S)]+\sqrt{\frac{\log(1/\delta)}{2m}}.$$

*Step 2: symmetrization.* Let $S'=\{(x_i',y_i')\}_{i=1}^m$ be an independent ghost sample and let $\sigma_1,\ldots,\sigma_m$ be independent Rademacher signs. Jensen's inequality and the symmetry of the joint law of $(S,S')$ under independent swaps give

$$
\begin{aligned}
\mathbb{E}_S[\Phi(S)]
&=\mathbb{E}_S\Big[\sup_{h\in\mathcal{H}}\big|\mathbb{E}_{S'}\big[\widehat{R}_{S'}(h)-\widehat{R}_S(h)\big]\big|\Big] \\
&\le\mathbb{E}_{S,S'}\Big[\sup_{h\in\mathcal{H}}\Big|\frac1m\sum_{i=1}^m\big(\ell(y_i',h(x_i'))-\ell(y_i,h(x_i))\big)\Big|\Big] \\
&=\mathbb{E}_{S,S',\sigma}\Big[\sup_{h\in\mathcal{H}}\Big|\frac1m\sum_{i=1}^m\sigma_i\big(\ell(y_i',h(x_i'))-\ell(y_i,h(x_i))\big)\Big|\Big] \\
&\le 2\,\mathbb{E}_{S,\sigma}\Big[\sup_{h\in\mathcal{H}}\Big|\frac1m\sum_{i=1}^m\sigma_i\,\ell(y_i,h(x_i))\Big|\Big].
\end{aligned}
$$

*Step 3: Massart's lemma.* Condition on $S$. For each $h$, let $a^h:=(\ell(y_1,h(x_1)),\ldots,\ell(y_m,h(x_m)))\in[0,1]^m$. As $h$ ranges over $\mathcal{H}$, the vectors $a^h$ take at most $\tau_{\mathcal{H}}(m)$ distinct values, and each has Euclidean norm at most $\sqrt m$. Since the supremum in Step 2 is inside an absolute value, apply Massart's lemma (Lemma 7.3.3) to the symmetrized set $A:=\{a^h:h\in\mathcal{H}\}\cup\{-a^h:h\in\mathcal{H}\}$, which has at most $2\tau_{\mathcal{H}}(m)$ elements:

$$\mathbb{E}_\sigma\Big[\sup_{h\in\mathcal{H}}\Big|\frac1m\sum_{i=1}^m\sigma_i a_i^h\Big|\Big]=\mathbb{E}_\sigma\Big[\max_{a\in A}\frac1m\sum_{i=1}^m\sigma_i a_i\Big]\le\sqrt{\frac{2\log\big(2\tau_{\mathcal{H}}(m)\big)}{m}}.$$

Hence $\mathbb{E}_S[\Phi(S)]\le 2\sqrt{2\log(2\tau_{\mathcal{H}}(m))/m}=\sqrt{8\log(2\tau_{\mathcal{H}}(m))/m}$.

*Step 4: assembling the constants.* Combining Steps 1 and 3 and using $(a+b)^2\le2a^2+2b^2$,

$$
\Phi(S)\le\sqrt{\frac{8\log(2\tau_{\mathcal{H}}(m))}{m}}+\sqrt{\frac{\log(1/\delta)}{2m}}
\le\sqrt{\frac{16\log(2\tau_{\mathcal{H}}(m))+\log(1/\delta)}{m}},
$$

which is the claim. $\square$

The bound is stated with $\tau_{\mathcal{H}}(m)$ rather than $\tau_{\mathcal{H}}(2m)$. The proof uses the growth function at $m$ because the symmetrization is applied directly to the $m$-sample; the ghost sample in Theorem 6.4.9 was counted through restrictions on $2m$ points, which is why that earlier bound involved $\tau_{\mathcal{H}}(2m)$. Since $\tau_{\mathcal{H}}$ is nondecreasing, replacing $m$ by $2m$ only weakens the bound.

**Corollary 8.6.7 (PAC risk bound for the Naive Bayes class).** Let $d_{\mathrm{N}}:=(C-1)(d+1)$ and $m\ge d_{\mathrm{N}}$. For every distribution $\mathcal{D}$ and every $\delta\in(0,1)$, with probability at least $1-\delta$, simultaneously for every $h\in\mathcal{H}_{\mathrm{NB}}$,

$$R(h)\le\widehat{R}_S(h)+\sqrt{\frac{16d_{\mathrm{N}}\log\big(2e(C-1)m/d_{\mathrm{N}}\big)+\log(1/\delta)}{m}}.$$

***Proof.*** By Proposition 8.6.2 and Theorem 8.6.4, $d_{\mathrm{N}}(\mathcal{H}_{\mathrm{NB}})\le d_{\mathrm{N}}$. Lemma 8.6.5 with $m\ge d_{\mathrm{N}}$ gives

$$\tau_{\mathcal{H}_{\mathrm{NB}}}(m)\le\Big(\frac{e(C-1)m}{d_{\mathrm{N}}}\Big)^{d_{\mathrm{N}}}.$$

Since $d_{\mathrm{N}}\ge1$,

$$\log\big(2\tau_{\mathcal{H}_{\mathrm{NB}}}(m)\big)\le\log2+d_{\mathrm{N}}\log\frac{e(C-1)m}{d_{\mathrm{N}}}\le d_{\mathrm{N}}\log\frac{2e(C-1)m}{d_{\mathrm{N}}}.$$

Substituting into Theorem 8.6.6 gives the claim. $\square$

For brevity, denote the deviation term in Corollary 8.6.7 by

$$\epsilon_m(\delta):=\sqrt{\frac{16d_{\mathrm{N}}\log\big(2e(C-1)m/d_{\mathrm{N}}\big)+\log(1/\delta)}{m}}.$$

The bound is distribution-free: it holds for every $\mathcal{D}$ on $\mathcal{X}\times[C]$, whether or not the conditional independence assumption is true. In the regime of interest the logarithmic factor is negligible, and the deviation scale is

$$\epsilon_m(\delta)=O\!\left(\sqrt{\frac{(C-1)(d+1)\log\big(Cm/((C-1)(d+1))\big)+\log(1/\delta)}{m}}\right).$$

## The Naive Bayes MLE and ERM over the Same Class

We now apply the uniform bound to the two estimators studied in this chapter. The first is the maximum-likelihood estimator. Under the Naive Bayes factorization, the likelihood decouples across classes and features, and its maximizer is given by the per-class frequency counts

$$\widehat\pi_c=\frac{m_c}{m},\quad \widehat\theta_{j,c}=\frac{\sum_{i:\,y_i=c}x_{ij}}{m_c},\quad m_c:=\sum_{i:\,y_i=c}1.$$

This is the estimator analyzed earlier in the chapter. When every count is positive, it lies in the interior of the parameter space and therefore induces a member of $\mathcal{H}_{\mathrm{NB}}$. When some count is zero, the corresponding log-score is infinite; the standard remedy is add-$\alpha$ smoothing,

$$\widetilde\theta_{j,c}=\frac{\sum_{i:\,y_i=c}x_{ij}+\alpha}{m_c+2\alpha},\qquad \widetilde\pi_c=\frac{m_c+\alpha}{m+C\alpha},\qquad \alpha>0,$$

which is also what the implementation below uses. The risk bound applies to any data-dependent member of $\mathcal{H}_{\mathrm{NB}}$, so it covers both the interior MLE and its smoothed version.

**Corollary 8.6.8 (Risk bound for the Naive Bayes MLE).** Let $\widehat\eta_{\mathrm{MLE}}$ be any maximum-likelihood estimator of the Naive Bayes parameters whose coordinates lie in the interior of the parameter space, or let it be the add-$\alpha$ smoothed estimator. Then, with probability at least $1-\delta$,

$$R\big(h_{\widehat\eta_{\mathrm{MLE}}}\big)\le\widehat{R}_S\big(h_{\widehat\eta_{\mathrm{MLE}}}\big)+\epsilon_m(\delta).$$

***Proof.*** The estimator is a data-dependent member of $\mathcal{H}_{\mathrm{NB}}$, and Corollary 8.6.7 is uniform over the class. $\square$

Corollary 8.6.8 is the missing risk bound for the vanilla Naive Bayes estimator. It is a *generalization* guarantee: it compares the true and empirical risks of the estimator actually returned, but it does not compare that estimator with the best hypothesis in $\mathcal{H}_{\mathrm{NB}}$. The latter is the content of the next result.

The second estimator minimizes the empirical zero-one risk over the same class:

$$\widehat h_{\mathrm{ERM}}\in\arg\min_{h\in\mathcal{H}_{\mathrm{NB}}}\ \widehat{R}_S(h).$$

This is a different optimization problem from maximum likelihood. The likelihood objective is a product of per-class and per-feature terms and decomposes into independent one-dimensional estimation problems; the zero-one objective couples all features and all classes through the argmax that defines $h_\eta$. In general, $h_{\widehat\eta_{\mathrm{MLE}}}$ is not an empirical risk minimizer over $\mathcal{H}_{\mathrm{NB}}$. Both estimators nevertheless live in the same hypothesis class, so the same capacity term controls both.

**Theorem 8.6.9 (ERM oracle inequality over the Naive Bayes class).** Let $\widehat h_{\mathrm{ERM}}\in\arg\min_{h\in\mathcal{H}_{\mathrm{NB}}}\widehat{R}_S(h)$. Then, with probability at least $1-\delta$,

$$R(\widehat h_{\mathrm{ERM}})\le\inf_{h\in\mathcal{H}_{\mathrm{NB}}}R(h)+2\epsilon_m(\delta).$$

***Proof.*** On the event of Corollary 8.6.7, for every $h\in\mathcal{H}_{\mathrm{NB}}$,

$$R(\widehat h_{\mathrm{ERM}})\le\widehat{R}_S(\widehat h_{\mathrm{ERM}})+\epsilon_m(\delta)\le\widehat{R}_S(h)+\epsilon_m(\delta)\le R(h)+2\epsilon_m(\delta),$$

where the middle inequality uses the definition of $\widehat h_{\mathrm{ERM}}$. Taking the infimum over $h\in\mathcal{H}_{\mathrm{NB}}$ gives the claim. $\square$

If the infimum in Theorem 8.6.9 is not attained, the same argument applies to any $\epsilon$-approximate empirical risk minimizer, by adding $\epsilon$ to the right-hand side. The theorem is a *learning* guarantee: it says that ERM over $\mathcal{H}_{\mathrm{NB}}$ is agnostic PAC learnable relative to that class. It does not say that ERM is computationally efficient. Minimizing the zero-one loss over $\mathcal{H}_{\mathrm{NB}}$ is a nonconvex combinatorial problem in general, whereas maximum likelihood is closed-form. This is the precise sense in which the two estimators trade statistical optimality for computational tractability.

The two guarantees are summarized in the following table.

| | Naive Bayes MLE | ERM over $\mathcal{H}_{\mathrm{NB}}$ |
|:--|:--|:--|
| objective | $\sum_i\log P_\eta(x_i,y_i)$ | $\sum_i\mathds{1}\{h_\eta(x_i)\ne y_i\}$ |
| computation | closed-form per-class counts | coupled; nonconvex in general |
| guarantee | $R\le\widehat{R}_S+\epsilon_m(\delta)$ | $R\le\inf_{\mathcal{H}_{\mathrm{NB}}}R+2\epsilon_m(\delta)$ |
| role | the estimator actually used | oracle benchmark and learning guarantee |

Theorem 8.6.9 compares ERM with the best predictor *inside* $\mathcal{H}_{\mathrm{NB}}$, not with the Bayes classifier. The two benchmarks coincide exactly when the true conditional distribution factorizes as $P(x\mid y)=\prod_jP(x_j\mid y)$: then the Bayes classifier is a member of $\mathcal{H}_{\mathrm{NB}}$, so the infimum in Theorem 8.6.9 equals the Bayes risk, and the bound controls the excess risk over the best possible predictor. When the distribution does not factorize, the infimum can be strictly larger than the Bayes risk; the difference is the approximation error of the Naive Bayes model, and Theorem 8.6.9 controls only the estimation error around that restricted optimum. In neither case does the *bound itself* require the factorization: Corollary 8.6.7 and Theorem 8.6.9 are distribution-free, and the factorization enters only in the interpretation of the approximation error.

The parameter-level Hoeffding bound derived earlier and the classifier-level bound of this section are complementary. The former controls the estimation error of each conditional probability and explains why the Naive Bayes estimates are stable; the latter controls the risk of the decision rule and explains why the classifier generalizes. Neither implies the other: parameter errors can compensate in the argmax, and a small risk does not require the individual probabilities to be accurate. What the capacity analysis adds is the correct dependence on the *number of decision rules*, which for Naive Bayes is governed by $(C-1)(d+1)$ rather than by the $Cd+C-1$ parameters or by the exponentially many entries of a full contingency table.

Two extensions are immediate. For categorical features with $V_j$ values, one-hot encoding replaces $d$ by $\sum_j(V_j-1)$ and the same bound holds with this effective dimension. For Gaussian Naive Bayes with a shared covariance matrix across classes, the log-score is affine in $x$, so the analysis applies unchanged; with class-specific covariance matrices the score is quadratic, the relevant Natarajan dimension is of order $Cd^2$, and the rate changes accordingly. These variants are left as exercises.

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

Despite the crudeness of median binarization (which discards all information about *how far* a measurement is from the median) and the naive independence assumption across all $30$ features, the classifier reaches over $90\%$ test accuracy with a training procedure that amounts to frequency counting — no gradient descent, no iterative optimization, just the closed-form MLE/MAP formulas of this chapter applied once per class.
