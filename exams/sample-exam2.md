
The exam contains 13 questions. All questions ask for an evaluation of five statements with (yes/no) or (true/false) answers. A true answer adds $+1$ points and a false answer adds $-1$ points to your exam score. Hence, skipping to evaluate a statement is very likely to be more advantageous for you than making a random guess. Your final score is calculated as (Your point sum)$/(13 \times 5) \times 100$. The exam duration is four hours.

**Notation.** The conventions are those of the lecture notes. $P(\cdot)$ denotes a probability — of an event, or of a discrete random variable taking a value; lowercase $p(\cdot)$ is reserved for the density of a continuous random variable. $\mathbb{E}[\cdot]$, $\mathrm{Var}[\cdot]$ and $\mathrm{Cov}[\cdot,\cdot]$ denote expectation, variance and covariance, with subscripts naming the variable averaged over. $\mathbb{1}(\cdot)$ is the indicator function and $\log$ the natural logarithm. $S$ is a training set of size $m$, drawn i.i.d. from a data distribution $\mathcal{D}$ over $\mathcal{X}\times\mathcal{Y}$; $d$ is the feature dimension and $C$ the number of classes. $\mathcal{H}$ is a hypothesis class, $h$ a hypothesis, $A$ a learning algorithm, $\ell(y,\widehat{y})$ the pointwise loss (true label first), $R(h)$ the generalization error and $\widehat{R}_S(h)$ the empirical error. $\top$ denotes matrix or column-vector transpose. All random variables are assumed to have finite expectations.

**No assumptions may be made additional to those specified in the question text.** No question requires a numerical calculation; where a quantitative claim appears, it is the *scaling behaviour* or the *direction* of an effect that is at stake, never an arithmetic value.

---

## Question 1

A team is asked to deploy a predictor for a task in which labels are expensive and the data-collection protocol is under their own control. They debate what they are and are not entitled to conclude from the framework of Chapter 1. Evaluate the following statements.

**Answer 1.1:** Committing to a hypothesis class $\mathcal{H}$ before seeing the data is a restriction that can only hurt: a learner free to search all measurable functions would be at least as good, provided it minimizes the empirical risk exactly.

**Answer 1.2:** The memorization hypothesis of Chapter 1 achieves zero empirical risk on every training set, which shows that a small empirical risk is not by itself evidence of a small generalization error.

**Answer 1.3:** If the protocol is changed so that the same subject is measured repeatedly over time, the i.i.d. assumption underlying every bound in the course is violated, and those bounds do not automatically transfer to the new setting.

**Answer 1.4:** Once the training set is large enough, the generalization error $R(h)$ of a fixed hypothesis $h$ can be computed exactly from the data.

**Answer 1.5:** The uniform-convergence bounds of Chapter 5 remain valid no matter which hypothesis of $\mathcal{H}$ the learning algorithm returns, including one chosen adversarially after the bound has been computed.

---

## Question 2

An engineer must certify, before deployment, that the true risk of a system exceeds its measured training risk by no more than a stated amount, with a stated confidence. Evaluate the following statements about the concentration inequalities available for this purpose.

**Answer 2.1:** Chebyshev's inequality requires a finite variance while Hoeffding's inequality requires the random variables to be bounded; neither requires knowing the shape of the underlying distribution.

**Answer 2.2:** Applying Markov's inequality to $e^{sX}$ rather than to $X$ itself is what converts a polynomially decaying tail bound into an exponentially decaying one, at the price of assuming a finite moment generating function.

**Answer 2.3:** Hoeffding's inequality, applied to a hypothesis fixed in advance, remains valid when that hypothesis is instead selected by minimizing the empirical risk on the very same sample.

**Answer 2.4:** McDiarmid's inequality can be applied to the supremum of the generalization gap over a hypothesis class because replacing a single training example moves that supremum by at most $1/m$, and this holds however many hypotheses the class contains.

**Answer 2.5:** Because the uniform Hoeffding bound degrades only logarithmically in the number of hypotheses monitored, doubling the size of a finite hypothesis class roughly doubles the sample size needed to retain the same accuracy and confidence.

---

## Question 3

A group models a binary outcome with a Bernoulli likelihood and must decide between maximum likelihood, maximum a-posteriori, and full Bayesian treatment, and later extends the model to many features. Evaluate the following statements.

**Answer 3.1:** Maximum a-posteriori estimation avoids computing the evidence because the evidence does not depend on the parameter, and the resulting predictor nevertheless performs model averaging over the posterior.

**Answer 3.2:** Conjugacy is a property of a prior-likelihood pair, and its practical value is that the posterior update reduces to a closed-form change of hyperparameters instead of an integral.

**Answer 3.3:** In the Beta-Bernoulli model the influence of the prior on the posterior predictive vanishes as the sample size grows, so the Bayesian and maximum likelihood predictions agree in the large-sample limit.

**Answer 3.4:** The naive Bayes assumption states that the features are independent of one another, so observing that two features are correlated in the data rules the classifier out.

**Answer 3.5:** The benefit of the naive Bayes assumption that its PAC analysis makes precise is that the number of parameters to estimate falls from exponential to linear in the number of features, so the sample size sufficient for a given per-parameter accuracy grows only logarithmically in that number.

---

## Question 4

A practitioner tunes a linear model on a data set whose features are measured on very different scales, and compares an $L_2$-penalized fit against an $L_1$-penalized one. Evaluate the following statements.

**Answer 4.1:** Ridge regression admits a closed-form solution while Lasso does not, because the $L_1$ penalty fails to be differentiable exactly where its solutions tend to lie.

**Answer 4.2:** The reason ridge regression remains solvable when $Z^\top Z$ is singular is that adding $\lambda I$ makes the matrix invertible; this is a numerical side effect, unrelated to the statistical purpose of the penalty.

**Answer 4.3:** Computing the z-score normalization constants from the whole data set rather than from the training split alone leaks information from the test split, and therefore makes the test error an optimistically biased estimate of the generalization error.

**Answer 4.4:** A model trained with an $L_2$ penalty produces sparse weight vectors in which most coordinates are exactly zero, whereas an $L_1$ penalty only shrinks coordinates towards zero without reaching it.

**Answer 4.5:** Structural risk minimization differs from empirical risk minimization in that it minimizes an upper bound on the generalization error rather than the empirical error itself, which is why its penalty term must depend on the hypothesis and not only on the data.

---

## Question 5

A model is reported to have a training error very close to its test error, and the team must decide what to change next. Evaluate the following statements about the two decompositions of the risk.

**Answer 5.1:** The approximation-estimation decomposition is pure bookkeeping and therefore holds for any loss function, whereas the additive bias-variance decomposition is specific to the squared loss.

**Answer 5.2:** The Bayes error is a property of the data distribution alone: neither collecting more data, nor enlarging the hypothesis class, nor changing the algorithm can reduce it.

**Answer 5.3:** For a convex hypothesis class, the squared bias term is at most the approximation error.

**Answer 5.4:** If a hypothesis class is not convex, the average predictor can have strictly lower risk than every single member of that class; this is the mechanism random forests exploit.

**Answer 5.5:** A model whose training error equals its test error is, by that fact alone, free of both approximation error and estimation error.

---

## Question 6

Evaluate the following statements about PAC learnability and the generalization bound for a finite hypothesis class.

**Answer 6.1:** The finite-class bound is distribution-free — it holds for every data distribution — which is precisely why it must express a worst case and can be very loose for a benign distribution.

**Answer 6.2:** Agnostic PAC learnability is a stronger requirement on the learner than realizable PAC learnability, because the agnostic learner must succeed for every distribution rather than only for those a member of the class fits perfectly.

**Answer 6.3:** In the agnostic setting, halving the target accuracy $\epsilon$ at fixed confidence roughly doubles the sample size the bound requires.

**Answer 6.4:** In the binary classification setting of the Fundamental Theorem, a class can be agnostic PAC learnable while no empirical risk minimization rule succeeds on it.

**Answer 6.5:** Occam's razor enters the finite-class bound as the fact that, at equal empirical risk, the bound prefers the hypothesis drawn from the smaller class.

---

## Question 7

A colleague claims to have found an algorithm that outperforms all competitors, having benchmarked it on twelve public data sets. Evaluate the following statements, in the light of the No Free Lunch theorem and the theory of shattering.

**Answer 7.1:** The No Free Lunch theorem is compatible with the existence of an algorithm that works well on every data set arising in one particular application domain, because it only asserts the existence of a hard distribution, not that such a distribution is one anybody would encounter.

**Answer 7.2:** A class shatters a set of points if it can realize every possible labeling of them, so a class of VC dimension $d$ shatters every set of $d$ points.

**Answer 7.3:** Since the VC dimension is defined by a worst case over point sets, two classes with the same VC dimension necessarily have the same growth function.

**Answer 7.4:** The Sauer-Shelah-Perles lemma converts a qualitative statement — that no set of $d+1$ points is shattered — into a quantitative one — that the number of realizable labelings grows only polynomially in the sample size — and it is that polynomial-versus-exponential gap that makes the subsequent bounds non-vacuous.

**Answer 7.5:** The class of unconstrained decision trees has infinite VC dimension and is therefore not PAC learnable, while restricting it to trees with at most $L$ leaves restores PAC learnability at a sample complexity that grows with $L$.

---

## Question 8

Evaluate the following statements about the Fundamental Theorem of Statistical Learning and the nonuniform learning framework built on top of it.

**Answer 8.1:** The Fundamental Theorem states an equivalence between six properties, so establishing any single one of them — exhibiting a finite VC dimension, for instance — yields the other five.

**Answer 8.2:** The theorem also fixes the constants of the sample complexity, so two classes of equal VC dimension require exactly the same number of samples for the same $(\epsilon,\delta)$.

**Answer 8.3:** The realizable and agnostic sample complexities in the theorem differ in their dependence on $\epsilon$ — $1/\epsilon$ against $1/\epsilon^2$ — so when the class contains a perfect hypothesis, far fewer samples suffice for a target accuracy.

**Answer 8.4:** Nonuniform learnability is a strictly weaker requirement than agnostic PAC learnability, obtained by letting the sufficient sample size depend on the hypothesis being competed against.

**Answer 8.5:** In the Minimum Description Length rule the weights over hypotheses are chosen by the learner after inspecting the training set, so that short descriptions are assigned to the hypotheses that happen to fit the data well.

---

## Question 9

Evaluate the following statements about Rademacher complexity and the machinery that surrounds it.

**Answer 9.1:** Empirical Rademacher complexity is computable from the sample alone, with no knowledge of the data distribution, which is what allows it to produce data-dependent bounds that VC dimension cannot.

**Answer 9.2:** A class rich enough to realize every labeling of the sample has empirical Rademacher complexity near its maximum, while a class consisting of a single hypothesis has empirical Rademacher complexity exactly zero.

**Answer 9.3:** Because Rademacher complexity is defined through a supremum over the class, adding to the class a hypothesis that the learning algorithm would never select leaves the resulting bound unchanged.

**Answer 9.4:** The contraction lemma is what transfers a bound stated for the hypothesis class to the loss composed with it, and its constant is the Lipschitz constant of that loss — which is why a bound stated at margin threshold $\nu$ carries a factor $1/\nu$.

**Answer 9.5:** VC dimension and Rademacher complexity are two independent theories of generalization, so a bound proved with one cannot in general be recovered from the other.

---

## Question 10

Evaluate the following statements about covering numbers, chaining, and PAC-Bayes bounds.

**Answer 10.1:** A covering-number bound is obtained by applying the finite-class bound to a finite proxy for the class and paying an additional term for the approximation that proxy introduces; the resolution $\epsilon$ of the cover is a free parameter to be balanced against the size of the cover.

**Answer 10.2:** Because the covering numbers of a $d$-dimensional ball grow exponentially in $d$, covering-number bounds are vacuous in high dimension.

**Answer 10.3:** Chaining sharpens a covering-number bound by summing it over a geometric sequence of resolutions instead of fixing a single one, which is what removes a residual logarithmic factor from the rate.

**Answer 10.4:** In a PAC-Bayes bound the prior must be fixed before the training set is seen, whereas the posterior may depend on the data in any way at all, including being the output of the learning algorithm.

**Answer 10.5:** The PAC-Bayes bound speaks only about genuinely randomized predictors, so it cannot be used to say anything about a single deterministic hypothesis.

---

## Question 11

A team grows decision trees on a data set with noisy labels, then aggregates many of them. Evaluate the following statements.

**Answer 11.1:** The greedy tree-growing algorithm is a tractable surrogate for empirical risk minimization over the tree class, and it is not guaranteed to return the empirical risk minimizer among trees of the size it produces.

**Answer 11.2:** Measuring entropy in bits rather than in nats can change which split the greedy algorithm selects at a node.

**Answer 11.3:** Pruning against a held-out validation set is a data-driven approximation of the same structural risk minimization trade-off that the leaf-budget bound expresses in closed form.

**Answer 11.4:** Bagging reduces the variance term of the excess risk while leaving the bias term unchanged, and its benefit is largest for base learners whose output is sensitive to the particular sample, such as deep decision trees.

**Answer 11.5:** Because bootstrap samples are all drawn from a single training set, the members of a bagged ensemble are statistically independent, and the variance of the ensemble is therefore exactly $1/T$ of a single member's.

---

## Question 12

A practitioner runs AdaBoost with decision stumps, observes that the training error reaches zero after a few dozen rounds, and must decide whether to stop there. Evaluate the following statements.

**Answer 12.1:** A $\gamma$-weak learner is required to beat random guessing by $\gamma$ on every reweighting of the training set, not merely on the uniform weighting, and it is this uniformity that AdaBoost consumes.

**Answer 12.2:** AdaBoost's weight $w_t$ for a weak hypothesis is a heuristic choice; any other monotone function of $\epsilon_t$ would yield the same exponential decay of the training error.

**Answer 12.3:** Once the training error reaches zero, further rounds cannot change the ensemble's predictions on the training set, and therefore cannot improve its generalization guarantee either.

**Answer 12.4:** The VC dimension of the class of $T$-term linear combinations of a base class grows roughly linearly in $T$, so a bound built on it predicts that boosting past the point of a perfect fit should degrade test performance — a prediction that practice contradicts.

**Answer 12.5:** The margin bound escapes that prediction because the Rademacher complexity of the convex hull of a base class equals the Rademacher complexity of the base class itself, however many members are combined.

---

## Question 13

A network with far more weights than training examples is observed to generalize well, and a colleague proposes replacing it with a kernel method. Evaluate the following statements.

**Answer 13.1:** For a network with more weights than training examples the VC-dimension bound of Chapter 5 is vacuous, yet the network can still generalize; the resolution offered in the notes is a complexity measure controlled by weight norms rather than by a count of parameters.

**Answer 13.2:** The efficiency of backpropagation rests on reusing the pre-activations already computed during the forward pass, which is what makes the cost of one gradient evaluation linear rather than quadratic in the number of weights.

**Answer 13.3:** The norm-based Rademacher bound for a two-layer network certifies that gradient descent will find a hypothesis that generalizes well.

**Answer 13.4:** The representer theorem holds for an arbitrary data-fit term together with any strictly increasing function of the RKHS norm as the regularizer, which is why it covers the squared loss of kernel ridge regression and the hinge loss of the support vector machine alike.

**Answer 13.5:** Because the reproducing kernel Hilbert space of the RBF kernel is infinite-dimensional, fitting kernel ridge regression requires solving an optimization problem whose number of free parameters grows with the dimension of that space.

---

# Answer Key

*(To be removed from the version handed to students.)*

| | .1 | .2 | .3 | .4 | .5 |
|---|---|---|---|---|---|
| **Q1** | False | True | True | False | True |
| **Q2** | True | True | False | True | False |
| **Q3** | False | True | True | False | True |
| **Q4** | True | False | True | False | True |
| **Q5** | True | True | False | True | False |
| **Q6** | True | True | False | False | True |
| **Q7** | True | False | False | True | True |
| **Q8** | True | False | True | True | False |
| **Q9** | True | True | False | True | False |
| **Q10** | True | False | True | True | False |
| **Q11** | True | False | True | True | False |
| **Q12** | True | False | False | True | True |
| **Q13** | True | True | False | True | False |

## Notes on the intended reasoning

**1.1** False. The No Free Lunch theorem shows the class of all functions $\mathcal{X}\to\{0,1\}$ is not PAC learnable on an infinite domain; the memorization hypothesis attains $\widehat R_S = 0$ while learning nothing. **1.4** False: $R(h)$ depends on the unknown $\mathcal{D}$ and can only be bounded or estimated, never computed.

**2.3** False. Hoeffding's inequality presumes $h$ fixed independently of $S$; a data-selected hypothesis requires a union bound over the whole class, which is exactly the content of Theorem 5.1. **2.5** False: the bound depends on $\log|\mathcal{H}|$, so doubling $|\mathcal{H}|$ adds $\log 2$ inside a square root — a negligible change, not a doubling of $m$.

**3.1** False. MAP is a point estimate and performs no model averaging; that is precisely what distinguishes it from the posterior predictive. **3.4** False: the assumption is *conditional* independence given the class, and the classifier remains effective in practice when it fails.

**4.2** False. The invertibility and the shrinkage are two faces of the same term: Chapter 9 shows $\lambda$ caps $\|w^*\|_2$ and hence the Rademacher complexity of the class the solution comes from. **4.4** False: the roles of $L_1$ and $L_2$ are interchanged.

**5.3** False. For convex $\mathcal{H}$ the inequality runs the other way, $\text{Bias}^2 \geq \epsilon_{app}$, with $\text{Variance} \leq \mathbb{E}_S[\epsilon_{est}]$. **5.5** False: matching train and test error indicates small *estimation* error only; a badly underfitting model has both errors equally large.

**6.3** False. The agnostic sample complexity scales as $1/\epsilon^2$, so halving $\epsilon$ multiplies $m$ by about four. **6.4** False: the Fundamental Theorem makes "agnostic PAC learnable" and "every ERM rule succeeds" equivalent in this setting.

**7.2** False. Finite VC dimension $d$ means *some* set of $d$ points is shattered; most sets of that size need not be. **7.3** False: the VC dimension is a single number and does not determine the growth function.

**8.2** False. The theorem bounds the sample complexity between $C_1$ and $C_2$ multiples of the same expression, leaving both constants unspecified. **8.5** False: the weights must be fixed before $S$ is seen, since a union bound over a data-dependent allocation is invalid — this is the same discipline the PAC-Bayes prior obeys.

**9.3** False. The supremum can only grow when a hypothesis is added, and the bound is algorithm-independent by construction — this is exactly the weakness that algorithmic stability was introduced to address. **9.5** False: Theorem 5a.3 derives the VC bound from Sauer-Shelah-Perles plus Massart's lemma, exhibiting VC dimension as one way of bounding Rademacher complexity.

**10.2** False. It is $\log N(\epsilon)$ that enters the bound, and it grows only linearly in $d$. **10.5** False: taking a Dirac posterior recovers the finite-class bound of Theorem 5.1 for deterministic hypotheses.

**11.2** False. Changing the logarithm's base rescales the entropy by a positive constant, and a positive rescaling cannot change which split maximizes the gain. **11.5** False: bootstrap samples share the same underlying $S$, so the members are correlated and the variance reduction is weaker than the idealized $1/T$.

**12.2** False. $w_t = \tfrac12\log\frac{1-\epsilon_t}{\epsilon_t}$ is the exact minimizer of the per-round factor $(1-\epsilon_t)e^{-w}+\epsilon_t e^{w}$; the proof of the exponential decay uses precisely this. **12.3** False: the exponential loss keeps decreasing, driving the margins up, which is what the margin bound rewards.

**13.3** False. The bound certifies only that a hypothesis with small weight norms generalizes; it gives the optimizer no credit for finding one. **13.5** False: by the representer theorem the solution is a kernel expansion with $m$ coefficients, one per training example, whatever the dimension of the RKHS.
