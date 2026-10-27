Solving hard real-world problems with machine learning demands advanced algorithmic designs. These designs should be based on a solid understanding of the principles that determine the behavior of learning algorithms. Statistical learning theory provides mathematical tools to develop such an understanding. It seeks answers to fundamental questions about the nature of learning from data such as

 - Under which conditions can a learning algorithm generalize from the training data to new data?
 - How can we measure the complexity of a learning task?
 - How can we control the complexity of a learning algorithm?
 - How can we estimate the generalization performance of a learning algorithm?

Every generalization bound in this course, no matter how it is derived, has the same shape: with high probability, the gap $R(h) - \widehat{R}_S(h)$ between true and empirical risk, uniformly over $h \in \mathcal{H}$, is controlled by a single number that measures **how complex $\mathcal{H}$ is**. What changes between different theories of generalization is only *how that number is defined* — as a raw count ($|\mathcal{H}|$, this chapter), a combinatorial shattering capacity ($d_{VC}(\mathcal{H})$, this chapter), an average best-case correlation with random noise (**Rademacher complexity**), a description length in a fixed metric (**covering numbers**), or a KL-divergence budget between a data-dependent posterior and a fixed prior over hypotheses (**PAC-Bayes bounds**) — the last three being the subject of Chapter 7. This chapter builds the foundational machinery — the finite-hypothesis-class bound, PAC learnability, the No Free Lunch theorem, VC dimension, and the Fundamental Theorem of Statistical Learning — using the coarsest and most concrete of these complexity measures, $|\mathcal{H}|$ and $d_{VC}(\mathcal{H})$. Chapter 7 then shows that these are two special cases of one abstract "complexity of a hypothesis class" idea, of which Rademacher complexity, covering numbers, and PAC-Bayes bounds are three more.

Building on what we covered in the probability theory chapter, let us start by investigating the last of the four questions above.

---

# Generalization Bounds

Remember the basic definitions below.

**Definition 6.1.1 (Generalization error)** Given a hypothesis $h \in \mathcal{H}$, a loss function $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow [0,1]$, and a data distribution $(x,y) \sim \mathcal{D}$, the generalization error of $h$ is defined as

$$R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(y, h(x))].$$

---

**Definition 6.1.2 (Empirical error)** Given a hypothesis $h \in \mathcal{H}$, a loss function $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow [0,1]$, and a data set $S = \{(x_1,y_1),...,(x_m,y_m)\}$, the empirical error of $h$ is defined as

$$\widehat{R}_S(h) = \frac{1}{m} \sum_{i=1}^m \ell(y_i, h(x_i)).$$

---

Intuitively, $R(h)$ is the error we actually care about but can never observe directly, since it averages over the whole (unknown) data distribution $\mathcal{D}$; $\widehat{R}_S(h)$ is the only quantity we can compute, since it averages over the finite sample $S$ we happen to have. The rest of this chapter is about how well the second quantity stands in for the first.

Using the definitions above and the concentration inequalities we covered in the probability theory chapter, we can derive the following theorem.

**Theorem 6.1.3 (Generalization bound)** Let $\mathcal{H}$ be a finite hypothesis set. Then for any $\delta \in (0,1)$, with probability at least $1-\delta$ over the choice of $S \sim \mathcal{D}^m$, for all $h \in \mathcal{H}$

$$R(h) \leq \widehat{R}_S(h) + \sqrt{\frac{\log |\mathcal{H}| + \log \frac{1}{\delta}}{2m}}.$$

**Proof.** Let $\mathcal{H} = \{h_1,...,h_{|\mathcal{H}|}\}$.  Then we have

$$
\begin{aligned}
P_{S \sim \mathcal{D}^m}\Big(\exists h_i \in \mathcal{H}: R(h_i) - \widehat{R}_S(h_i) > \epsilon \Big)
&=P_{S \sim \mathcal{D}^m}\Big(\textstyle\bigcup_{i=1}^{|\mathcal{H}|} \big \{ R(h_i) - \widehat{R}_S(h_i) > \epsilon \big \} \Big) \\
&\leq \sum_{i=1}^{|\mathcal{H}|} P_{S \sim \mathcal{D}^m}\big(  R(h_i) - \widehat{R}_S(h_i) > \epsilon\big) \\
&\leq  |\mathcal{H}| \exp(-2m\epsilon^2).
\end{aligned}
$$
where the first inequality follows from the union bound and the second from Hoeffding's inequality (Theorem 5.6.5) applied to each $h_i$ separately, with the loss bounded in $[0,1]$. Setting the right hand side to $\delta$ and solving for $\epsilon$ yields the final result $\square$

---

This inequality is called a generalization bound, because it bounds the generalization error of any hypothesis $h \in \mathcal{H}$ in terms of its empirical error. We can derive a number of interesting consequences from this bound:
 - The bound is uniform in the sense that it holds simultaneously for all hypotheses in $\mathcal{H}$.
 - The bound is independent of the data distribution $\mathcal{D}$ and the loss function $\ell$.
 - The bound increases logarithmically with the size of the hypothesis set $|\mathcal{H}|$, that is, a richer hypothesis set is more likely to overfit.
 - The bound decreases with the size of the training set $m$, that is, more data is less likely to overfit.
 - For a fixed training set size $m$ and two different hypothesis sets $\mathcal{H}_1$ and $\mathcal{H}_2$, the bound prefers the smaller hypothesis set. This is known as the **Occam's razor principle**. The principle was introduced by the 14th century theologian William of Ockham. It states that among competing hypotheses, the one with the fewest assumptions should be selected.
 - The bound gives a **Probably Approximately Correct (PAC)** performance guarantee. The event that any hypothesis in $\mathcal{H}$ is **approximately correct** in the sense that its generalization error is at most $\epsilon$ with probability at least $1-\delta$.

---

# PAC Learnability

The formal notion below is due to [Valiant (1984)](https://doi.org/10.1145/1968.1972)<!-- cite: valiant1984theory | article | author={Valiant, Leslie G.}; title={A Theory of the Learnable}; journal={Communications of the ACM}; year={1984}; volume={27}; number={11}; pages={1134--1142} -->, who founded the **Probably Approximately Correct (PAC)** framework this whole chapter is built on.

**Definition 6.2.1 (Realizability).** A hypothesis space $\mathcal{H}$ is realizable with respect to a loss $\ell$ and data distribution $\mathcal{D}$ if $\exists h^* \in \mathcal{H}$ such that $R(h^*) = 0$.

---

In words, realizability says the target concept itself is expressible inside $\mathcal{H}$, with no approximation error to worry about — the only obstacle between the learner and a perfect predictor is a finite sample. It trivially follows from this definition that $\min_{h \in \mathcal{H}} \widehat{R}_S(h)=0$  with probability 1 for a sample set $S$ collected i.i.d. from $\mathcal{D}$. Hence, under the realizability assumption, all ERM solutions $h_S \in \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h)$ give zero error.

---

**Definition 6.2.2 (Representativeness)** A training set $S$ is called $\epsilon-$ representative if

$$\forall h \in \mathcal{H}, |R(h) - \widehat{R}_S(h)| \leq \epsilon.$$

---

In words, on an $\epsilon$-representative sample, training error is a faithful proxy for true error, simultaneously for *every* hypothesis in $\mathcal{H}$ — not just the one a particular algorithm happens to output.

---

**Lemma 6.2.3** Assume that a training set $S$ is $\frac{\epsilon}{2}$-representative, then any ERM solution $h_S \in \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h)$ satisfies

$$R(h_S) \leq \min_{h \in \mathcal{H}} R(h) + \epsilon.$$

***Proof.*** For every $h \in \mathcal{H}$,
$R(h_S) \le \widehat{R}_S(h_S) + \frac{\epsilon}{2} \le \widehat{R}_S(h) + \frac{\epsilon}{2} \le R(h) + \frac{\epsilon}{2} + \frac{\epsilon}{2} = R(h) + \epsilon$,
where the first and last steps use $\frac{\epsilon}{2}$-representativeness and the middle step uses that $h_S$ minimizes $\widehat{R}_S$. Taking the minimum over $h \in \mathcal{H}$ on the right gives the claim $\square$ 

In words: on a representative sample, ERM cannot be fooled — the hypothesis it picks by minimizing *empirical* risk is also near-optimal in *true* risk, because empirical and true risk cannot disagree by more than $\epsilon/2$ for any hypothesis, including both $h_S$ and the true best one.

---

**Definition 6.2.4 (Uniform Convergence).** A hypothesis class $\mathcal{H}$ has the **uniform convergence property** if there exists a function $m_{\mathcal{H}}^{\text{UC}} : (0, 1)^2 \to \mathbb{N}$ such that for every $\epsilon, \delta \in (0, 1)$ and every distribution $\mathcal{D}$, any sampled data set $S=\{(x_i,y_i) \overset{i.i.d.}{\sim} D : i = 1, \ldots, m\}$ with $m \ge m_{\mathcal{H}}^{\text{UC}}(\epsilon, \delta)$ is $\epsilon$-representative with probability at least $1-\delta$.

---

In words, $\mathcal{H}$ has uniform convergence if, given enough samples, a random data set is (with high probability) $\epsilon$-representative — so that empirical risk minimization is safe to use on it, no matter which distribution $\mathcal{D}$ generated the data.

---

**Definition 6.2.5 (Agnostic PAC Learnability)** A hypothesis class $\mathcal{H}$ is **agnostic PAC learnable** if there exist a function $m_{\mathcal{H}}: (0, 1)^2 \to \mathbb{N}$ and a **learning algorithm** $A$ such that for every $\epsilon, \delta \in (0, 1)$ and every distribution $\mathcal{D}$, running the learning algorithm on data set $S=\{(x_i,y_i) \overset{i.i.d.}{\sim} D : i = 1, \ldots, m\}$ with $m \ge m_{\mathcal{H}}(\epsilon, \delta)$ satisfies

$$P\big(\{S \sim \mathcal{D}^m: R(A(S)) \le \min_{h' \in \mathcal{H}} R(h') + \epsilon\}\big) \geq 1-\delta.$$

---

In words, an agnostic PAC learner is guaranteed, given enough data, to output a hypothesis nearly as good as the best one $\mathcal{H}$ has to offer — for *any* data distribution $\mathcal{D}$, and without assuming $\mathcal{H}$ contains a perfect predictor (hence "agnostic").

---

**Corollary 6.2.6** If $|\mathcal{H}| < \infty$ (finite hypothesis class), then it has the uniform convergence property with sample complexity

$$m_{\mathcal{H}}^{\text{UC}}(\epsilon, \delta) \le \left\lceil \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2} \right\rceil.$$

and is agnostically PAC learnable using the ERM algorithm with sample complexity

$$m_{\mathcal{H}}(\epsilon, \delta) \le m_{\mathcal{H}}^{\text{UC}}(\epsilon/2, \delta) \le \left\lceil \frac{2\log(2|\mathcal{H}|/\delta)}{\epsilon^2} \right\rceil.$$

***Proof.*** Definition 6.2.2 asks for a *two-sided* bound, so run the proof of Theorem 6.1.3 over the $2|\mathcal{H}|$ bad events $\{R(h_i) - \widehat{R}_S(h_i) > \epsilon\}$ and $\{\widehat{R}_S(h_i) - R(h_i) > \epsilon\}$, $i = 1,\ldots,|\mathcal{H}|$. The union bound and Hoeffding's inequality (Theorem 5.6.5, applied to each tail separately) give
$$P\Big(\exists h \in \mathcal{H}: |R(h)-\widehat{R}_S(h)| > \epsilon\Big) \leq 2|\mathcal{H}|\, e^{-2m\epsilon^2}.$$
Requiring the right-hand side to be at most $\delta$ and solving for $m$ yields $m \geq \log(2|\mathcal{H}|/\delta)/(2\epsilon^2)$, i.e. any such $m$ makes $S$ $\epsilon$-representative with probability at least $1-\delta$: this is the first claim. For the second, Lemma 6.2.3 turns $\epsilon/2$-representativeness into the agnostic PAC guarantee for ERM, so it suffices to take $m \geq m_{\mathcal{H}}^{\text{UC}}(\epsilon/2,\delta) = \log(2|\mathcal{H}|/\delta)/(2(\epsilon/2)^2) = 2\log(2|\mathcal{H}|/\delta)/\epsilon^2$ $\square$

In short: finiteness of $\mathcal{H}$ alone is already enough to guarantee learnability, with the required sample size growing only *logarithmically* in $|\mathcal{H}|$ — doubling the hypothesis class costs only one extra bit of sample complexity.

---

**Definition 6.2.7 (Bayes Error)** Given a data distribution $(x,y) \sim \mathcal{D}$, the **Bayes error** is the risk of the best possible predictor over *all* measurable functions $\mathcal{X} \rightarrow \mathcal{Y}$ — not merely the best one inside some hypothesis class $\mathcal{H}$:

$$R^*_{\mathcal{D}} := \inf_{h': \mathcal{X} \rightarrow \mathcal{Y}} R(h').$$

---

A predictor $h^*$ that achieves the Bayes error is called a **Bayes (optimal) predictor**. For binary classification with $\mathcal{Y} = \{0,1\}$, the Bayes predictor is

$$h^*(x) = \arg \max_{y \in \{0,1\}} P(y \mid x),$$

with Bayes error obtained by averaging the pointwise error over the input distribution,

$$R^*_{\mathcal{D}} = \mathbb{E}_{x}\Big[1 - \max_{y \in \{0,1\}} P(y \mid x)\Big] = \mathbb{E}_{x}\Big[\min_{y \in \{0,1\}} P(y \mid x)\Big];$$

***Proof (of the Bayes-error formula).*** For any predictor $h$ and any $x$, the conditional risk under the zero-one loss is

$$P\big(h(x) \neq y \mid x\big) = 1 - P\big(y = h(x) \mid x\big) \geq 1 - \max_{y \in \{0,1\}} P(y \mid x) = \min_{y \in \{0,1\}} P(y \mid x).$$

Taking expectations over $x$ gives $R(h) \geq \mathbb{E}_x[\min_{y} P(y \mid x)]$ for every $h$. Conversely, fix a measurable tie-breaking rule and let $h^*(x)$ be the selected maximizer of $P(\cdot \mid x)$; then $P(y = h^*(x) \mid x) = \max_y P(y \mid x)$ at every $x$, so $R(h^*) = \mathbb{E}_x[\min_{y} P(y \mid x)]$. Since $R^*_{\mathcal{D}}$ is the infimum of $R(h)$ over all measurable predictors, the displayed identity follows. The same argument for a finite label set gives $R^*_{\mathcal{D}} = \mathbb{E}_x[1 - \max_{y} P(y \mid x)]$. $\square$

This floor is also called the **noise** level of $\mathcal{D}$, since it is the part of the risk that no predictor — however expressive — can remove. It is zero exactly when the label is a deterministic function of the input.

For any predictor $h$, the difference $R(h) - R^*_{\mathcal{D}}$ is called the **excess risk** of $h$: the part of the risk that a better predictor could, in principle, remove.

---

The Bayes error $R^*_{\mathcal{D}}$ is a property of the distribution $\mathcal{D}$ alone. It should not be confused with $\min_{h \in \mathcal{H}} R(h)$, the best risk achievable *within a fixed, possibly restricted* hypothesis class $\mathcal{H}$ — which we call the **approximation error** of $\mathcal{H}$ below. In general $\min_{h \in \mathcal{H}} R(h) \geq R^*_{\mathcal{D}}$, with equality only when $\mathcal{H}$ happens to contain a Bayes predictor for $\mathcal{D}$.

---

**Definition 6.2.8 ("Realizable" PAC learnability)** A hypothesis class $\mathcal{H}$ is **PAC learnable** if there exist a function $m_{\mathcal{H}}: (0,1)^2 \to \mathbb{N}$ and a learning algorithm $A$ such that for every $\epsilon, \delta \in (0,1)$ and every distribution $\mathcal{D}$ satisfying the realizability assumption (Definition 6.2.1), running $A$ on $S = \{(x_i,y_i) \overset{i.i.d.}{\sim} \mathcal{D} : i=1,\ldots,m\}$ with $m \geq m_{\mathcal{H}}(\epsilon,\delta)$ satisfies

$$P\big(\{S \sim \mathcal{D} : R(A(S)) \leq \epsilon\}\big) \geq 1-\delta.$$

---

Realizability forces $\min_{h \in \mathcal{H}} R(h) = 0$, and in particular $R^*_{\mathcal{D}} = 0$. Definition 6.2.8 is therefore a *weaker demand on the learner* than Definition 6.2.5: it is only required to succeed on the restricted family of distributions for which some hypothesis in $\mathcal{H}$ is perfect, whereas an agnostic learner must succeed on **every** distribution. Consequently, agnostic PAC learnability trivially implies PAC learnability — simply apply Definition 6.2.5 to a realizable $\mathcal{D}$, where $\min_{h'\in\mathcal{H}}R(h') = 0$ makes the two guarantees read identically.

What is far from trivial is that, for binary classification under the zero-one loss, the converse holds as well: the two notions turn out to be *equivalent*, and both are equivalent to $\mathcal{H}$ having a finite VC dimension. This is the content of the Fundamental Theorem of Statistical Learning (Theorem 6.4.7) below. The equivalence is specific to this setting; for general label spaces and loss functions the realizable and agnostic notions genuinely differ, and the sample complexities differ even in the binary case — $\Theta(1/\epsilon)$ in the realizable case against $\Theta(1/\epsilon^2)$ in the agnostic case, as Theorem 6.4.7 makes precise.

---

# Bias-Complexity Dilemma (Trade-off)

The result below is in the spirit of [Wolpert (1996)](https://doi.org/10.1162/neco.1996.8.7.1341)<!-- cite: wolpert1996lack | article | author={Wolpert, David H.}; title={The Lack of A Priori Distinctions Between Learning Algorithms}; journal={Neural Computation}; year={1996}; volume={8}; number={7}; pages={1341--1390} -->.

**Theorem 6.3.1 (No free lunch).** Let $A$ be any learning algorithm for the task of binary classification with respect to the 0/1 loss over a domain $\mathcal{X}$ and $m < |\mathcal{X}|/2$ be a positive integer representing a training set size. Then, there exists a distribution $\mathcal{D}$ over $\mathcal{X} \times \{0, 1\}$ such that:
 * There exists a function $f : \mathcal{X} \to \{0, 1\}$ with $R(f) = 0$.
 * With probability of at least $1/7$ over the choice of $S \sim \mathcal{D}^m$ we have that $R(A(S)) \ge 1/8$.

***Proof.*** Fix any $C \subseteq \mathcal{X}$ with $|C| = 2m$; this is possible since $m < |\mathcal{X}|/2$. There are exactly $T := 2^{2m}$ functions $f_1,\ldots,f_T : C \to \{0,1\}$. For each $i$, let $\mathcal{D}_i$ be the distribution on $\mathcal{X}\times\{0,1\}$ that draws $x$ uniformly from $C$ and sets $y = f_i(x)$ deterministically. By construction $R_{\mathcal{D}_i}(f_i) = 0$, so the first bullet holds for every $\mathcal{D}_i$. It remains to exhibit one $i$ for which $A$ fails.

*Step 1: reduce to an average.* We show
$$\max_{i \in [T]}\ \mathbb{E}_{S \sim \mathcal{D}_i^m}\big[R_{\mathcal{D}_i}(A(S))\big] \;\geq\; \tfrac14 .$$
Enumerate the $k := (2m)^m$ possible input sequences $S_1,\ldots,S_k$ drawable from $C$, and write $S_j^i$ for the sequence $S_j$ labeled by $f_i$. Since drawing $S \sim \mathcal{D}_i^m$ is exactly drawing one of the $S_j$ uniformly and labeling it by $f_i$,
$$
\begin{aligned}
\mathbb{E}_{S \sim \mathcal{D}_i^m}\big[R_{\mathcal{D}_i}(A(S))\big] &= \frac{1}{k}\sum_{j=1}^k R_{\mathcal{D}_i}\big(A(S_j^i)\big).
\end{aligned}
$$
A maximum is at least an average, and averages may be exchanged, so
$$
\begin{aligned}
\max_{i}\frac1k\sum_{j} R_{\mathcal{D}_i}(A(S_j^i)) &\;\geq\; \frac1T\sum_{i}\frac1k\sum_{j} R_{\mathcal{D}_i}(A(S_j^i)) \\
&= \frac1k\sum_{j}\frac1T\sum_{i} R_{\mathcal{D}_i}(A(S_j^i)) \;\geq\; \min_{j} \frac1T\sum_{i} R_{\mathcal{D}_i}(A(S_j^i)).
\end{aligned}
$$

*Step 2: bound the inner average for a fixed sequence.* Fix $j$, write $S_j = (x_1,\ldots,x_m)$, and let $v_1,\ldots,v_p$ be the elements of $C$ that **do not** appear in $S_j$. Since $S_j$ contains at most $m$ distinct elements and $|C| = 2m$, we have $p \geq m$. For any hypothesis $h$,
$$R_{\mathcal{D}_i}(h) = \frac{1}{2m}\sum_{x \in C} \mathds{1}\big(h(x) \neq f_i(x)\big) \;\geq\; \frac{1}{2m}\sum_{r=1}^p \mathds{1}\big(h(v_r) \neq f_i(v_r)\big) \;\geq\; \frac{1}{2p}\sum_{r=1}^p \mathds{1}\big(h(v_r) \neq f_i(v_r)\big),$$
the last step using $p \geq m$. Averaging over $i$ and exchanging the two finite averages,
$$\frac1T\sum_{i} R_{\mathcal{D}_i}(A(S_j^i)) \;\geq\; \frac{1}{2}\cdot\frac1p\sum_{r=1}^p \underbrace{\frac1T\sum_{i} \mathds{1}\big(A(S_j^i)(v_r) \neq f_i(v_r)\big)}_{=:\ \Pi_r} \;\geq\; \frac12 \min_{r} \Pi_r .$$

*Step 3: the unseen points are a coin flip.* Fix $r$ and pair up the $T$ functions: partition $[T]$ into $T/2$ pairs $(i,i')$ such that $f_i$ and $f_{i'}$ agree everywhere on $C$ **except** at $v_r$. Since $v_r$ does not occur in $S_j$, the two labeled sequences coincide, $S_j^i = S_j^{i'}$, so $A$ returns the *same* hypothesis $h$ on both. But $f_i(v_r) \neq f_{i'}(v_r)$, so $h$ disagrees with exactly one of them at $v_r$. Each pair therefore contributes exactly one to the sum, giving $\Pi_r = \tfrac12$ for every $r$. Chaining Steps 1-3 gives $\max_i \mathbb{E}_{S\sim\mathcal{D}_i^m}[R_{\mathcal{D}_i}(A(S))] \geq \tfrac12\cdot\tfrac12 = \tfrac14$.

*Step 4: from expectation to probability.* Let $\mathcal{D} := \mathcal{D}_{i^*}$ for a maximizing $i^*$ and let $\theta := R_{\mathcal{D}}(A(S)) \in [0,1]$, a random variable of the draw of $S$, with $\mathbb{E}[\theta] \geq \tfrac14$. Apply Markov's inequality (Theorem 5.6.1) to the non-negative variable $1-\theta$ with threshold $\tfrac78$:
$$P\big(\theta < \tfrac18\big) = P\big(1-\theta > \tfrac78\big) \leq \frac{\mathbb{E}[1-\theta]}{7/8} \leq \frac{3/4}{7/8} = \frac67,$$
hence $P(\theta \geq \tfrac18) \geq \tfrac17$ $\square$

The proof is worth re-reading for what it does *not* assume: nothing about $A$ beyond it being a function of the sample. Step 3 is the crux — on the at least $m$ points of $C$ the learner never saw, the label is, by construction of the family $\{f_i\}$, an independent fair coin flip that no amount of cleverness can predict.

---

**Corollary 6.3.2** Let $\mathcal{X}$ be an infinite domain ($|\mathcal{X}|=\infty$, e.g. a real-valued feature space) and $\mathcal{H}$ be the set of all functions from $\mathcal{X}$ to $\{0,1\}$. Then $\mathcal{H}$ is not PAC learnable.

***Proof (Shalev-Shwartz book).*** Assume, by way of contradiction, that the class is learnable. Choose some $\epsilon < 1/8$ and $\delta < 1/7$. By the definition of PAC learnability, there must be some learning algorithm $A$ and an integer $m = m(\epsilon, \delta)$, such that for any data-generating distribution over $\mathcal{X} \times \{0, 1\}$, if for some function $f : \mathcal{X} \to \{0, 1\}$, $R(f)=0$, then with probability greater than $1 - \delta$ when $A$ is applied to samples $S$ of size $m$, generated i.i.d. by $\mathcal{D}$, $R(A(S)) \le \epsilon$. However, applying the No-Free-Lunch theorem, since $|\mathcal{X}| > 2m$, for every learning algorithm (and in particular for the algorithm $A$), there exists a distribution $\mathcal{D}$ such that with probability greater than $1/7 > \delta$, $R(A(S)) > 1/8 > \epsilon$, which leads to the desired contradiction $\square$

---

The main conclusion of the theorem and its corollary is that there is no universal learner that works well for all data distributions. There always exists a task for which the learner fails. This is a very important result. It tells us that we need to make assumptions about the data distribution in order to design a good learner. These assumptions will constitute the **inductive bias** of the learner. The no free lunch theorem tells us only that we need to induce a degree of bias into the learner. However, it does not address its consequences. Inducing too much bias limits the ability of the learner to explain the training observations. Inducing too little bias leads to overfitting. The goal is to find the right balance between the two. This dilemma is known as the **bias-complexity dilemma**.

## Decomposing the Excess Risk

Let us describe the bias-complexity dilemma in more formal terms. Let $h_S \gets A(S)$ be the hypothesis a learning algorithm returns on a sample $S$ — for instance an ERM solution $h_S \in \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h)$. Definition 6.2.7 already identified an irreducible floor $R^*_{\mathcal{D}}$ that no predictor can undercut. Subtracting it, and adding and subtracting the best risk achievable inside $\mathcal{H}$, splits the generalization error into three terms:

$$\underbrace{R(h_S)}_{\text{generalization error}} = \underbrace{R^*_{\mathcal{D}}}_{\text{Bayes error (noise)}} + \underbrace{\min_{h \in \mathcal{H}} R(h) - R^*_{\mathcal{D}}}_{\epsilon_{app} \text{ (approximation error)}} + \underbrace{R(h_S) - \min_{h \in \mathcal{H}} R(h)}_{\epsilon_{est} \text{ (estimation error)}}.$$

Each term is controlled by a different factor, and only the last two are ours to influence:

* The **Bayes error** $R^*_{\mathcal{D}}$ is a property of $\mathcal{D}$ alone (Definition 6.2.7). No choice of $\mathcal{H}$, algorithm, or sample size touches it.
* The **approximation error** $\epsilon_{app}$ measures how badly $\mathcal{H}$ fails to contain a Bayes predictor. It is a property of $\mathcal{H}$ and $\mathcal{D}$, and does *not* depend on $S$: enlarging the class can only reduce it, since $\mathcal{H}' \supset \mathcal{H}$ implies $\min_{h' \in \mathcal{H}'} R(h') \leq \min_{h \in \mathcal{H}} R(h)$.
* The **estimation error** $\epsilon_{est}$ measures how much worse the hypothesis we actually found is than the best one available in $\mathcal{H}$. This is the only term the generalization bounds of this chapter control: representativeness (Definition 6.2.2) with parameter $\epsilon/2$ gives $\epsilon_{est} \leq \epsilon$ by Lemma 6.2.3, and Theorem 6.1.3 bounds it by a quantity growing in $|\mathcal{H}|$ and shrinking in $m$.

Herein lies the dilemma. Enlarging $\mathcal{H}$ drives $\epsilon_{app}$ down but $\epsilon_{est}$ up; shrinking it does the reverse. Since we only ever observe $\widehat{R}_S(h_S)$, and a richer class drives *that* towards zero regardless of what happens to $R(h_S)$, an unconstrained search inflates $\epsilon_{est}$ invisibly — which is exactly overfitting. This is the **bias-complexity dilemma**.

---

## Bias-Variance Decomposition

For the squared loss, the same excess risk admits a second, more classical split. Consider a regression problem with observations $(x,y) \sim \mathcal{D}$, and let $h_S \in \mathcal{H}$ be fitted by minimizing the mean squared error. For a fixed test point $x$, the expected squared error over both the noisy label $y$ and the training sample $S$ is

$$
\begin{aligned}
     \mathbb{E}_{S,  y|x} \Big [ \Big(y - h_{S}(x) \Big)^2 \Big ]&=\mathbb{E}_{S, y|x} \Big [ y^2 - 2 y h_{S}(x) + h_{S}(x)^2  \Big ]\\
    &=\mathbb{E}_{y|x}  [ y^2] + \mathbb{E}_{S}  [  h_{S}(x)^2  ] - 2 \mathbb{E}_{y|x}  [ y ] \mathbb{E}_{S}  [ h_{S}(x) ] \\
    &=\mathbb{E}_{y|x}  [ y]^2 + \mathrm{Var}_{y|x}[y] + \mathbb{E}_{S}  [  h_{S}(x)  ]^2 + \mathrm{Var}_{S}  [  h_{S}(x)  ]- 2 \mathbb{E}_{y|x}  [ y ]\mathbb{E}_{S}  [ h_{S}(x) ]\\
    &= \Big ( \underbrace{\mathbb{E}_{y|x}  [ y ] -\mathbb{E}_{S}  [ h_{S}(x) ]}_{\text{Estimator~Bias}} \Big )^2 + \underbrace{\mathrm{Var}_{S}  [  h_{S}(x)  ]}_{\text{Estimator~Variance}}+\underbrace{\mathrm{Var}_{y|x}[y]}_{\text{Label~noise~variance}}
\end{aligned}
$$

The second line uses that $y$ and $S$ are independent given $x$, and we note that $\mathbb{E}_{S|x}[h_S(x)] = \mathbb{E}_S[h_S(x)]$ and $\mathrm{Var}_{S|x}[h_S(x)] = \mathrm{Var}_S[h_S(x)]$ since $S$ is collected independently of the test point $x$.

---

**Example.** Take the $k$-nearest neighbours regressor,
$$h_S(x) = \frac{1}{k}\sum_{i=1}^k y_{(i)},$$
where $(x_{(1)},y_{(1)}),\ldots,(x_{(k)},y_{(k)})$ are the $k$ training points nearest to $x$. Writing $f^*(x) := \mathbb{E}[y|x]$ and assuming label noise of variance $\sigma^2$, the three terms read

$$
 \begin{aligned}
    \mathbb{E}_{S,y|x} \Big [ \Big(y - h_{S}(x) \Big)^2 \Big ]= \Bigg (f^*(x) - \mathbb{E}_{S} \bigg [ \frac{1}{k} \sum_{i=1}^k f^*(x_{(i)}) \bigg ] \Bigg )^2 + \frac{\sigma^2}{k} + \sigma^2 .
\end{aligned}
$$

---

The middle term follows because averaging $k$ independent noisy labels divides the noise variance by $k$. The two extremes are instructive:

   - $k=m$: every training point is a "neighbour", so $h_S$ predicts the global sample mean of the labels regardless of $x$. The variance term is at its smallest ($\sigma^2/m$), but the bias is large, since a constant cannot track how $x$ influences $y$. The model **underfits**.
   - $k=1$: $h_S$ predicts the label of the single nearest training point to $x$. The bias term is small (for dense enough data, $f^*(x_{(1)}) \approx f^*(x)$), but the variance is at its largest ($\sigma^2$), since the prediction rides on one noisy label. The model **overfits**: it incurs no error on $S$, while its test error depends entirely on how representative that particular $S$ was.

---

## The Two Decompositions Are the Same Decomposition

The two splits above look unrelated: one is stated in terms of risks of hypotheses, the other in terms of moments of a predicted value at a point. For the squared loss they are in fact two ways of cutting up one and the same quantity. Let $f^*(x) := \mathbb{E}[y|x]$ denote the Bayes predictor for the squared loss, write $\|g\|^2 := \mathbb{E}_{x}[g(x)^2]$ for the $L_2(\mathcal{D})$ norm on the input space, and let

$$\bar{h}(x) := \mathbb{E}_S[h_S(x)]$$

be the **average predictor**: the hypothesis we would obtain by averaging the outputs of our learning algorithm over all training samples of size $m$.

**Step 1 (excess risk is a squared distance).** Conditioning on $x$ and expanding $(y-h(x))^2 = \big((y-f^*(x)) + (f^*(x)-h(x))\big)^2$, the cross term vanishes because $\mathbb{E}[y - f^*(x)\mid x] = 0$, leaving

$$\begin{gathered}
R(h) = \|h - f^*\|^2 + \underbrace{\mathbb{E}_x\big[\mathrm{Var}[y|x]\big]}_{= \, R^*_{\mathcal{D}}}, \\
\text{hence}\quad R(h) - R^*_{\mathcal{D}} = \|h-f^*\|^2 .
\end{gathered}$$

So for the squared loss, excess risk over the Bayes error *is* squared distance to the Bayes predictor, and the label-noise term of the bias-variance decomposition is exactly the Bayes error of Definition 6.2.7.

**Step 2 (average over samples).** Taking $\mathbb{E}_S$ of Step 1 and splitting $h_S - f^* = (h_S - \bar h) + (\bar h - f^*)$, whose cross term vanishes by the definition of $\bar h$,

$$\mathbb{E}_S\big[R(h_S)\big] - R^*_{\mathcal{D}} = \underbrace{\|\bar h - f^*\|^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_x\big[\mathrm{Var}_S[h_S(x)]\big]}_{\text{Variance}},$$

which is the bias-variance decomposition of the previous section, integrated over the test point $x$.

**Step 3 (the identity).** The very same left-hand side is, by the approximation-estimation decomposition,

$$\mathbb{E}_S\big[R(h_S)\big] - R^*_{\mathcal{D}} = \epsilon_{app} + \mathbb{E}_S[\epsilon_{est}].$$

Comparing Steps 2 and 3 gives the conceptual bridge we were after:

$$\boxed{\ \text{Bias}^2 + \text{Variance} \;=\; \epsilon_{app} + \mathbb{E}_S[\epsilon_{est}] \;=\; \mathbb{E}_S[R(h_S)] - R^*_{\mathcal{D}}\ }$$

Both decompositions carve up the *expected excess risk over the Bayes error*. They agree on the total and on the noise floor; they differ only in where they draw the internal line.

**Where the line differs.** Suppose $\mathcal{H}$ is convex, so that the average predictor $\bar h$ is itself a member of $\mathcal{H}$. Then

$$\begin{gathered}
\text{Bias}^2 = \|\bar h - f^*\|^2 \;\geq\; \min_{h \in \mathcal{H}}\|h-f^*\|^2 = \epsilon_{app}, \\
\text{and consequently}\quad \text{Variance} \;\leq\; \mathbb{E}_S[\epsilon_{est}].
\end{gathered}$$

In words: the bias term is the approximation error *plus* whatever part of the estimation error is systematic — the offset that survives averaging over samples. The variance term is the remaining, purely fluctuating part. The correspondence between the two vocabularies is therefore

| plays the role of | approximation-estimation | bias-variance (squared loss) |
|---|---|---|
| irreducible noise | $R^*_{\mathcal{D}}$ | $\mathbb{E}_x[\mathrm{Var}[y \mid x]]$ |
| class too restricted | $\epsilon_{app}$ | $\text{Bias}^2$ (which is $\geq \epsilon_{app}$) |
| finite sample | $\mathbb{E}_S[\epsilon_{est}]$ | $\text{Variance}$ (which is $\leq \mathbb{E}_S[\epsilon_{est}]$) |
| grows when $\mathcal{H}$ grows | $\epsilon_{est}$ | Variance |
| shrinks when $\mathcal{H}$ grows | $\epsilon_{app}$ | $\text{Bias}^2$ |

**Two caveats, both instructive.**

* *Convexity matters.* If $\mathcal{H}$ is not convex, $\bar h$ need not lie in $\mathcal{H}$, and it can then be strictly closer to $f^*$ than any single member of $\mathcal{H}$ — so $\text{Bias}^2 < \epsilon_{app}$ becomes possible. This is not a pathology to be avoided but a mechanism to be exploited: it is precisely why averaging many predictors trained on resampled data can beat every individual predictor, which is the theoretical content of **bagging** in Chapter 12.
* *The exact identity is specific to the squared loss.* Step 1 relied on the squared loss to turn excess risk into a squared distance, and Step 2 relied on that distance being a variance. For the zero-one loss no analogous *additive* bias-variance decomposition exists — the various proposals in the literature are definition-dependent and do not decompose additively. The approximation-estimation decomposition, by contrast, is pure bookkeeping (add and subtract $\min_{h\in\mathcal{H}}R(h)$ and $R^*_{\mathcal{D}}$) and therefore holds for **any** loss function whatsoever. This is why statistical learning theory is built on the approximation-estimation split rather than on bias-variance, even though the latter is the more familiar of the two.

---

# Vapnik - Chervonenkis (VC) Dimension

The generalization bound developed above was for a finite hypothesis set. However, in practice we often deal with infinite hypothesis sets. In order to develop generalization bounds for them, we need a measure of complexity that stays finite even when $|\mathcal{H}| = \infty$. [Vapnik and Chervonenkis (1971)](https://doi.org/10.1137/1116025)<!-- cite: vapnik1971uniform | article | author={Vapnik, V. N. and Chervonenkis, A. Ya.}; title={On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities}; journal={Theory of Probability and its Applications}; year={1971}; volume={16}; number={2}; pages={264--280} --> built such a measure on a remarkable observation: in a binary classification problem, a hypothesis set can report only a finite number of distinct labelings of a finite sample, even though it contains infinitely many hypotheses. The chain of definitions runs restriction $\rightarrow$ shattering $\rightarrow$ growth function $\rightarrow$ VC dimension; let us take them in turn.

---

**Definition 6.4.1 (Restriction)** Given $S = \{x_1, \ldots, x_m \} \subset \mathcal{X}$, the following set

$$\mathcal{H}_S = \{ (h(x_1), \ldots, h(x_m)) : h \in \mathcal{H} \}$$

is called a **restriction** of $\mathcal{H}$ to $S$.

---

In words, a restriction is the set of all unique label vectors a hypothesis class $\mathcal{H}$ can generate from a data set $S$.

---

**Definition 6.4.2 (Shattering)** $\mathcal{H}$ is said to **shatter** $S$ if $|\mathcal{H}_S|= 2^{|S|}$.

---

In words,  a hypothesis class $\mathcal{H}$ shatters a data set $S$ if the restriction of $\mathcal{H}$ to $S$ is the set of all functions from $S$ to $\{0,1\}$.

---

**Corollary 6.4.3** If a data set $C \subset \mathcal{X}$ with $2m$ data points (i.e. $|C|=2m$) is shattered by $\mathcal{H}$, then for every learning algorithm $A$, there exists a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, supported on $C$, such that some $h \in \mathcal{H}$ satisfies $R(h)=0$, yet $P\big(\{S' \sim \mathcal{D}^m : R(A(S')) \geq 1/8\}\big) \geq 1/7$.

***Proof.*** Run the proof of Theorem 6.3.1 with $\mathcal{X}$ replaced by $C$ itself, which is legitimate because $|C| = 2m > m$. That proof produced a distribution $\mathcal{D} = \mathcal{D}_{i^*}$, supported on $C$ and labeled by one of the $2^{2m}$ functions $f_{i^*}: C \to \{0,1\}$, on which $A$ fails as claimed. The only remaining point to check is that the perfect predictor may be taken *inside* $\mathcal{H}$: since $\mathcal{H}$ shatters $C$, its restriction $\mathcal{H}_C$ is all of $\{0,1\}^C$ (Definition 6.4.2), so **every** $f_i$ — $f_{i^*}$ in particular — agrees on $C$ with some $h \in \mathcal{H}$, and that $h$ has $R(h) = 0$ because $\mathcal{D}$ is supported on $C$ $\square$

In words, if $\mathcal{H}$ is large enough to shatter a data set of $2m$ on a domain $\mathcal{X}$, then we cannot find a learning algorithm as good as we desire (arbitrarily good) that can be run on data sets of size $m$ or smaller. The intuitive explanation is that due to the No Free Lunch theorem, under such circumstances, the hypothesis space $\mathcal{H}$ is so large that for any $2m$ data points, the true labels of the half that is not used in training can be perfectly explained by a hypothesis inside $\mathcal{H}$. This is a manifestation of Karl Popper's falsifiability principle in machine learning theory. This principle sets a demarcation point between science and other types of knowledge generation. An explanation is scientific if there exists a set of observations that can falsify it.

---

**Definition 6.4.4 (Growth function)** The growth function $\tau_{\mathcal{H}}: \mathbb{N}^+ \rightarrow \mathbb{N}^+$ of $\mathcal{H}$ is defined as

$$\tau_{\mathcal{H}}(m) = \max_{S \subseteq \mathcal{X},\, |S| = m} |\mathcal{H}_S|.$$

---

In words, the growth function counts the maximum number of distinct labelings that $\mathcal{H}$ can report on any data set on domain $\mathcal{X}$ that contains $m$ data points. 

Note that $\mathcal{H}$ shatters some $S \subseteq \mathcal{X}$ with $|S|=m$ if and only if $\tau_{\mathcal{H}}(m)=2^m$.

---

**Definition 6.4.5 (VC dimension)** The VC dimension of a hypothesis set $\mathcal{H}$ is the size of the largest data set that $\mathcal{H}$ can shatter:

$$d_{VC}(\mathcal{H}) = \sup \{m: \tau_{\mathcal{H}}(m) = 2^m\},$$

where the supremum is $\infty$ if $\mathcal{H}$ shatters arbitrarily large sets.

---

Intuitively, $d_{VC}(\mathcal{H})$ is the largest number of points $\mathcal{H}$ can label in *every* conceivable way — a single number summarizing how much a class can "memorize" arbitrary data, and hence how easily it can overfit; it plays exactly the role $\log|\mathcal{H}|$ played for finite classes, but stays finite even when $\mathcal{H}$ itself is infinite. Let us now see what a finite VC dimension provides — first qualitatively, then quantitatively.

---

**Theorem 6.4.6 (VC and PAC learnability).** If $d_{VC}(\mathcal{H})=\infty$ then $\mathcal{H}$ is not PAC learnable.

***Proof.*** For any training set size $m$, the infinite VC dimension allows $\mathcal{H}$ to shatter a data set of size $2m$. Then the result follows from Corollary 6.4.3 $\square$

This theorem is an analytical way of the argumentation presented above after the No Free Lunch theorem. Learning is only possible with some inductive bias. The bias-complexity dilemma stems from this obligation.

---

**Theorem 6.4.7 (The Fundamental Theorem of Statistical Learning).** Let $\mathcal{H}$ be a class of binary classifiers under the zero-one loss, and assume that $d_{VC}(\mathcal{H}) = d < \infty$. Then, there exist $C_1, C_2 \in \mathbb{R}^+$ such that for all $\epsilon,\delta \in (0,\tfrac1{16})$

* $\mathcal{H}$ has the uniform convergence property with sample complexity

    $$C_1 \frac{d + \log(1/\delta)}{\epsilon^2} \le m_{\mathcal{H}}^{\text{UC}}(\epsilon, \delta) \le C_2 \frac{d + \log(1/\delta)}{\epsilon^2}$$

* $\mathcal{H}$ is agnostic PAC learnable with sample complexity
    $C_1 \frac{d + \log(1/\delta)}{\epsilon^2} \le m_{\mathcal{H}}(\epsilon, \delta) \le C_2 \frac{d + \log(1/\delta)}{\epsilon^2}$

* $\mathcal{H}$ is PAC learnable with sample complexity (the lower bound for $d \geq 2$)

    $$C_1 \frac{d + \log(1/\delta)}{\epsilon} \le m_{\mathcal{H}}(\epsilon, \delta) \le C_2 \frac{d \log(1/\epsilon) + \log(1/\delta)}{\epsilon}$$

---

In words, the following statements are equal:

* $\mathcal{H}$ has the uniform convergence property.
* Any ERM rule is a successful agnostic PAC learner for $\mathcal{H}$.
* $\mathcal{H}$ is agnostic PAC learnable.
* $\mathcal{H}$ is PAC learnable.
* Any ERM rule is a successful PAC learner for $\mathcal{H}$.
* $\mathcal{H}$ has a finite VC-dimension.

This outcome is very useful because it allows one to achieve all six of these properties by ensuring only one of them.

***Proof.*** We prove the theorem in two parts: the **equivalence of the six statements**, which is where the conceptual content lies and which we establish completely; and the **sample complexity rates**, which rest on tools developed in Chapter 7: chaining for the agnostic upper bound, and the two-point and Assouad methods for the lower bounds.

**Part 1: the equivalence.** It suffices to close the cycle

$$
\begin{aligned}
\text{finite } d_{VC} \ &\Rightarrow\ \text{uniform convergence} \ \Rightarrow\ \text{ERM is agnostic PAC} \\
&\Rightarrow\ \text{agnostic PAC learnable} \ \Rightarrow\ \text{PAC learnable} \ \Rightarrow\ \text{finite } d_{VC},
\end{aligned}
$$

after which every statement implies every other, and the two ERM statements follow because the implications that produce them exhibit ERM itself as the successful learner.

1. *Finite $d_{VC} \Rightarrow$ uniform convergence.* By Lemma 6.4.8, $\tau_{\mathcal{H}}(m) \leq (em/d)^d$ is polynomial in $m$, so $\log\tau_{\mathcal{H}}(2m)/m \to 0$. Theorem 6.4.9 below (whose proof uses only Lemma 6.4.8, symmetrization, and Massart's lemma) then bounds the representativeness gap by a quantity tending to $0$, which is exactly Definition 6.2.4.
2. *Uniform convergence $\Rightarrow$ ERM is an agnostic PAC learner.* Immediate from Lemma 6.2.3: take $m \geq m_{\mathcal{H}}^{\text{UC}}(\epsilon/2,\delta)$, so that $S$ is $\epsilon/2$-representative with probability $\geq 1-\delta$, on which event every ERM output satisfies $R(h_S) \leq \min_{h}R(h) + \epsilon$.
3. *ERM is an agnostic PAC learner $\Rightarrow$ agnostic PAC learnable.* By definition — a witness algorithm has been exhibited.
4. *Agnostic PAC learnable $\Rightarrow$ PAC learnable.* Already argued after Definition 6.2.8: a realizable $\mathcal{D}$ has $\min_{h}R(h)=0$, so Definition 6.2.5 specializes to Definition 6.2.8.
5. *PAC learnable $\Rightarrow$ finite $d_{VC}$.* This is the contrapositive of Theorem 6.4.6, which was proved from Corollary 6.4.3 and hence from the No Free Lunch theorem.

**Part 2: the upper bounds.** *Agnostic case.* The chaining bound (7.4) of Chapter 7 states that with probability at least $1-\delta$,
$$\sup_{h\in\mathcal{H}}\big|R(h)-\widehat{R}_S(h)\big| \;\leq\; 96\sqrt{\frac{d}{m}} + 3\sqrt{\frac{\log(4/\delta)}{2m}}.$$
Both terms are at most $\epsilon/2$ as soon as $m \geq 192^2\,d/\epsilon^2$ and $m \geq 18\log(4/\delta)/\epsilon^2$, and $\log(4/\delta) \leq 2\log(1/\delta)$ for $\delta \leq \tfrac14$, so $m_{\mathcal{H}}^{\text{UC}}(\epsilon,\delta) \leq C_2\frac{d+\log(1/\delta)}{\epsilon^2}$ with $C_2 = 192^2 + 36$. Part 1, step 2 turns this into the same rate for $m_{\mathcal{H}}(\epsilon,\delta) \leq m_{\mathcal{H}}^{\text{UC}}(\epsilon/2,\delta)$, at the price of a factor $4$ in $C_2$. (The more elementary route through Theorem 7.3.5, which bounds the Rademacher complexity by $2\sqrt{d\log(m+1)/m}$ using Lemma 6.4.8 and Massart's lemma alone, gives the same rate up to a factor $\log(1/\epsilon)$; chaining is what removes it.)

*Realizable case.* Let $\mathcal{D}$ be realizable, so that every ERM hypothesis has $\widehat{R}_S(h_S) = 0$, and let $B := \{\exists h \in \mathcal{H} : \widehat{R}_S(h) = 0,\ R(h) > \epsilon\}$. We show that for $m \geq 8/\epsilon$,
$$P(B) \;\leq\; 2\,\tau_{\mathcal{H}}(2m)\,2^{-\epsilon m/2}. \tag{6.1}$$
*Step 1: a ghost sample.* Let $S'$ be an independent sample of size $m$ and $B' := \{\exists h \in \mathcal{H} : \widehat{R}_S(h) = 0,\ \widehat{R}_{S'}(h) \geq \epsilon/2\}$. On $B$, fix a hypothesis $h_S$ witnessing $B$, chosen as a function of $S$ alone. Given $S$, $\widehat{R}_{S'}(h_S)$ is the mean of $m$ i.i.d. Bernoulli variables with mean $\mu := R(h_S) > \epsilon$, so Chebyshev's inequality (Theorem 5.6.2) gives
$$P\big(\widehat{R}_{S'}(h_S) < \tfrac\epsilon2 \,\big|\, S\big) \;\leq\; P\big(|\widehat{R}_{S'}(h_S) - \mu| > \tfrac\mu2 \,\big|\, S\big) \;\leq\; \frac{\mu(1-\mu)/m}{\mu^2/4} \;\leq\; \frac{4}{m\epsilon} \;\leq\; \frac12 .$$
Hence $P(B' \mid S) \geq \tfrac12$ on $B$, and $P(B') \geq \tfrac12 P(B)$.

*Step 2: random swaps.* As in the proof of Theorem 6.4.9, exchanging the $i$-th example of $S$ with the $i$-th example of $S'$, independently for each $i$ with probability $\tfrac12$, leaves the joint law of $(S,S')$ unchanged. So $P(B')$ equals the probability of $B'$ after the swaps, and it suffices to bound the latter conditionally on the $2m$ examples. On these examples $\mathcal{H}$ has at most $\tau_{\mathcal{H}}(2m)$ distinct behaviours. Fix one, and let $r$ be the number of pairs $i$ on which it errs on exactly one of the two examples. If it errs on both examples of some pair, it cannot have zero error on $S$ after any swap. Otherwise, $B'$ requires that all its mistakes land in $S'$ and that there are at least $\epsilon m/2$ of them, so $r \geq \epsilon m/2$, and each of the $r$ pairs must be swapped in the one favourable way, which happens with probability $2^{-r} \leq 2^{-\epsilon m/2}$. A union bound over the behaviours gives $P(B') \leq \tau_{\mathcal{H}}(2m)\,2^{-\epsilon m/2}$, and Step 1 gives (6.1).

*Step 3: solve for $m$.* For $2m \geq d$, Lemma 6.4.8 gives $\log\tau_{\mathcal{H}}(2m) \leq d\log(2em/d)$, and for any $a > 0$ the inequality $\log u \leq u - 1$ at $u = a\cdot 2em/d$ gives $d\log(2em/d) \leq 2eam - d(1 + \log a)$. With $a := \frac{\epsilon\log 2}{8e}$ this reads $d\log(2em/d) \leq \frac{\epsilon m\log2}{4} + d\log\frac{8}{\epsilon\log 2}$. The right-hand side of (6.1) is therefore at most $\delta$ as soon as
$$\frac{\epsilon m\log 2}{4} \;\geq\; d\log\frac{8}{\epsilon\log2} + \log\frac2\delta, \quad\text{i.e.}\quad m \;\geq\; \frac{4}{\epsilon\log 2}\Big(d\log\frac{8}{\epsilon\log 2} + \log\frac2\delta\Big),$$
which also implies $m \geq 8/\epsilon$ and $2m \geq d$. On the complement of $B$, every ERM hypothesis has $R(h_S) \leq \epsilon$, so $m_{\mathcal{H}}(\epsilon,\delta) \leq C_2\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}$ for $\epsilon,\delta \in (0,\tfrac1{16})$. The $\log(1/\epsilon)$ enters through $\log\tau_{\mathcal{H}}(2m) = O(d\log m)$. Theorem 6.4.14, at the end of this chapter, is the same argument for a finite class, where no ghost sample is needed. The double-sample argument is due to [Blumer et al. (1989)](https://doi.org/10.1145/76359.76371)<!-- cite: blumer1989learnability | article | author={Blumer, Anselm and Ehrenfeucht, Andrzej and Haussler, David and Warmuth, Manfred K.}; title={Learnability and the Vapnik-Chervonenkis Dimension}; journal={Journal of the ACM}; year={1989}; volume={36}; number={4}; pages={929--965} -->; see also Section 28.3 of [Shalev-Shwartz & Ben-David (2014)](https://doi.org/10.1017/CBO9781107298019)<!-- cite: shalev2014understanding | book | author={Shalev-Shwartz, Shai and Ben-David, Shai}; title={Understanding Machine Learning: From Theory to Algorithms}; publisher={Cambridge University Press}; year={2014} -->.

**Part 3: the lower bounds.** Let $C \subseteq \mathcal{X}$ be a set of size $d$ shattered by $\mathcal{H}$, and let $m < d/2$. Apply Theorem 6.3.1 with the domain taken to be $C$ (legitimate since $m < |C|/2$) and then Corollary 6.4.3's observation that shattering places the perfect predictor inside $\mathcal{H}$: for **every** learning algorithm $A$ there is a realizable distribution $\mathcal{D}$ on $C \times \{0,1\}$ with

$$P\big(\{S \sim \mathcal{D}^m : R(A(S)) \geq 1/8\}\big) \geq 1/7 .$$

Hence for any $\epsilon < 1/8$ and $\delta < 1/7$ no sample size below $d/2$ can suffice, i.e. $m_{\mathcal{H}}(\epsilon,\delta) \geq d/2$: the sample complexity must grow at least linearly in the VC dimension, in the realizable case and therefore also in the agnostic case. The full rates come from letting the labels on the shattered set be $\epsilon$-biased coins rather than deterministic, so that the learner must additionally *estimate* $d$ separate biases. This is carried out in the minimax section of Chapter 7, whose Theorem 7.9.5 and inequality (7.7) give, for $\epsilon,\delta \in (0,\tfrac1{16})$:
* agnostic: $m_{\mathcal{H}}(\epsilon,\delta) \geq \max\Big(\frac{d}{4096\,\epsilon^2},\ \frac{\log(1/(4\delta))}{16\,\epsilon^2}\Big)$;
* realizable, $d \geq 2$: $m_{\mathcal{H}}(\epsilon,\delta) \geq \max\Big(\frac{d-1}{16\,\epsilon},\ \frac{\log(1/(2\delta))}{4\,\epsilon}\Big)$.

Since $\log(1/(4\delta)) \geq \tfrac12\log(1/\delta)$ for $\delta \leq \tfrac1{16}$, $d - 1 \geq d/2$ for $d \geq 2$, and a maximum is at least the average, both are of the claimed form $C_1\frac{d+\log(1/\delta)}{\epsilon^2}$, respectively $C_1\frac{d+\log(1/\delta)}{\epsilon}$. Finally, $m_{\mathcal{H}}(2\epsilon,\delta) \leq m^{\text{UC}}_{\mathcal{H}}(\epsilon,\delta)$ by Part 1, step 2, so the agnostic lower bound transfers to $m^{\text{UC}}_{\mathcal{H}}$ with $C_1$ divided by $4$. The restriction $d \geq 2$ is necessary: for $\mathcal{H} = \{h, 1-h\}$ a single example identifies the target $\square$

---

The VC dimension gives a binary characterization of the hypothesis class. One may want to analyze the capacity of a hypothesis class in relation to an arbitrary data set size smaller or larger than its VC dimension. The following lemma, independently due to [Sauer (1972)](https://doi.org/10.1016/0097-3165(72)90019-2)<!-- cite: sauer1972density | article | author={Sauer, N.}; title={On the Density of Families of Sets}; journal={Journal of Combinatorial Theory, Series A}; year={1972}; volume={13}; number={1}; pages={145--147} --> and [Shelah (1972)](https://doi.org/10.2140/pjm.1972.41.247)<!-- cite: shelah1972combinatorial | article | author={Shelah, Saharon}; title={A Combinatorial Problem; Stability and Order for Models and Theories in Infinitary Languages}; journal={Pacific Journal of Mathematics}; year={1972}; volume={41}; number={1}; pages={247--261} -->, is instrumental in relating the growth function of a hypothesis class to the sample size.

**Lemma 6.4.8 (Sauer-Shelah-Perles).** If $d_{VC}(\mathcal{H}) = d < \infty$, then for all $m \geq d$, we have $\tau_{\mathcal{H}}(m) \leq \sum_{i=0}^d {m \choose i}$. In particular, for $m \geq d$, we have $\tau_{\mathcal{H}}(m) \leq \left(\frac{em}{d}\right)^d$.

***Proof.*** The lemma follows from a stronger, purely combinatorial claim that makes no reference to $d$ at all:

$$\begin{gathered}
\textbf{Claim.}\quad |\mathcal{H}_S| \;\leq\; \big|\{ B \subseteq S : \mathcal{H} \text{ shatters } B \}\big|, \\
\text{for every finite } S \subseteq \mathcal{X}.
\end{gathered}$$

*Proof of the claim, by induction on $m = |S|$.* For $m=1$, say $S = \{x_1\}$: if $|\mathcal{H}_S| = 1$ the right-hand side is at least $1$ (the empty set is shattered by any non-empty class), and if $|\mathcal{H}_S| = 2$ then $\mathcal{H}$ shatters both $\emptyset$ and $\{x_1\}$, so the right-hand side is $2$.

For the inductive step, let $S = \{x_1,\ldots,x_m\}$ and $S' := \{x_2,\ldots,x_m\}$. Split the labelings in $\mathcal{H}_S$ according to whether their restriction to $S'$ occurs with one or with both values of the first coordinate. Formally, set

$$\begin{gathered}
Y_0 := \mathcal{H}_{S'}, \\
Y_1 := \big\{ (y_2,\ldots,y_m) : (0,y_2,\ldots,y_m) \in \mathcal{H}_S \text{ and } (1,y_2,\ldots,y_m) \in \mathcal{H}_S \big\},
\end{gathered}$$

so that every element of $\mathcal{H}_S$ is obtained either as the unique extension of an element of $Y_0$, or as one of the two extensions of an element of $Y_1$; counting gives exactly

$$|\mathcal{H}_S| = |Y_0| + |Y_1|.$$

The induction hypothesis applied to $\mathcal{H}$ on $S'$ bounds the first term: $|Y_0| = |\mathcal{H}_{S'}| \leq |\{B \subseteq S' : \mathcal{H} \text{ shatters } B\}|$. For the second, define the subclass

$$\mathcal{H}' := \big\{ h \in \mathcal{H} : \exists h' \in \mathcal{H},\ h'(x_1) = 1-h(x_1) \text{ and } h'(x_i)=h(x_i) \ \forall i \geq 2 \big\},$$

i.e. those hypotheses whose labeling of $S'$ is realized with *both* labels of $x_1$. By construction $Y_1 = \mathcal{H}'_{S'}$, and crucially: **if $\mathcal{H}'$ shatters $B \subseteq S'$ then $\mathcal{H}'$, hence $\mathcal{H}$, shatters $B \cup \{x_1\}$** — because each of the $2^{|B|}$ labelings of $B$ is available in $\mathcal{H}'$ with both possible labels of $x_1$. Therefore, by the induction hypothesis applied to $\mathcal{H}'$ on $S'$,

$$
\begin{aligned}
|Y_1| = |\mathcal{H}'_{S'}| &\leq \big|\{B \subseteq S' : \mathcal{H}' \text{ shatters } B\}\big| \\
&= \big|\{B \subseteq S' : \mathcal{H}' \text{ shatters } B\cup\{x_1\}\}\big| \\
&\leq \big|\{B \subseteq S : x_1 \in B,\ \mathcal{H}\text{ shatters } B\}\big|.
\end{aligned}
$$

Adding the two bounds counts each shattered subset of $S$ exactly once — those avoiding $x_1$ in the first term, those containing it in the second — which proves the claim.

*From the claim to the lemma.* Since $d_{VC}(\mathcal{H}) = d$, no subset of size $> d$ is shattered, so the right-hand side of the claim counts at most all subsets of size at most $d$:
$$\tau_{\mathcal{H}}(m) = \max_{|S|=m} |\mathcal{H}_S| \leq \sum_{i=0}^{d}\binom{m}{i}.$$
For the closed form, assume $m \geq d$ so that $(d/m)^i \geq (d/m)^d$ for every $i \leq d$, and apply the binomial theorem:
$$\Big(\frac{d}{m}\Big)^{d}\sum_{i=0}^d \binom{m}{i} \;\leq\; \sum_{i=0}^d \binom{m}{i}\Big(\frac{d}{m}\Big)^{i} \;\leq\; \sum_{i=0}^m \binom{m}{i}\Big(\frac{d}{m}\Big)^{i} = \Big(1+\frac{d}{m}\Big)^m \leq e^{d},$$
the last step by $1+u \leq e^u$. Rearranging gives $\sum_{i=0}^d\binom{m}{i} \leq (em/d)^d$ $\square$

The lemma is the combinatorial heart of the whole theory: it converts a *binary* statement ("$\mathcal{H}$ cannot shatter $d+1$ points") into a *quantitative* one ("$\mathcal{H}$ produces at most polynomially many labelings"), and it is the polynomial-versus-exponential gap this opens that every subsequent bound exploits.

---

**Theorem 6.4.9 (Sample complexity wrt growth function).** Let $\mathcal{H}$ be a hypothesis class with growth function $\tau_{\mathcal{H}}$. Then for every distribution $\mathcal{D}$ and every $\delta \in (0,1)$, a sample set $S=\{(x_i,y_i) \overset{i.i.d.}{\sim} \mathcal{D} : i = 1, \ldots, m\}$ satisfies

$$P\left (\left \{S \sim \mathcal{D}^m : \sup_{h \in \mathcal{H}} |R(h) - \widehat{R}_S(h)| \leq \frac{1}{\delta}\sqrt{\frac{2\log\big(2\,\tau_{\mathcal{H}}(2m)\big)}{m}} \right \} \right ) \geq 1 - \delta.$$

***Proof.*** The proof has two steps: bound the *expected* representativeness gap, then convert to a high-probability statement by Markov's inequality.

*Step 1 (expectation bound).* Introduce a **ghost sample** $S' = \{(x_i',y_i')\}_{i=1}^m \sim \mathcal{D}^m$, independent of $S$. Since $R(h) = \mathbb{E}_{S'}[\widehat{R}_{S'}(h)]$, Jensen's inequality (an absolute value of an expectation is at most the expectation of the absolute value, and a supremum of expectations is at most the expectation of the supremum) gives

$$\mathbb{E}_S\Big[\sup_{h}\big|R(h)-\widehat{R}_S(h)\big|\Big] \;\leq\; \mathbb{E}_{S,S'}\Big[\sup_{h}\big|\widehat{R}_{S'}(h)-\widehat{R}_S(h)\big|\Big].$$

Because $S$ and $S'$ are i.i.d. and mutually independent, exchanging the $i$-th pair of $S$ with the $i$-th pair of $S'$ leaves the joint law unchanged. Recording the exchanges with independent Rademacher signs $\sigma_i \in \{-1,+1\}$ therefore changes nothing:

$$
\begin{aligned}
\mathbb{E}_{S,S'}\Big[\sup_h\Big|\frac1m\sum_i\big(\ell(y_i', h(x_i'))-\ell(y_i, h(x_i))\big)\Big|\Big] \\
= \mathbb{E}_{S,S',\sigma}\Big[\sup_h\Big|\frac1m\sum_i \sigma_i\big(\ell(y_i', h(x_i'))-\ell(y_i, h(x_i))\big)\Big|\Big].
\end{aligned}
$$

Now **condition on $S$ and $S'$** and look at the inner expectation over $\sigma$ alone. For each $h$ define the vector $a^{(h)} \in \mathbb{R}^m$ with $a_i^{(h)} := \frac1m\big(\ell(y_i', h(x_i')) - \ell(y_i, h(x_i))\big)$. Two observations:

* $a^{(h)}$ depends on $h$ only through $h$'s restriction to the $2m$ points $\{x_1,\ldots,x_m,x_1',\ldots,x_m'\}$, so at most $\tau_{\mathcal{H}}(2m)$ distinct vectors arise (Definition 6.4.4);
* each coordinate lies in $[-1/m,1/m]$ because the loss lies in $[0,1]$, hence $\|a^{(h)}\|_2 \leq \sqrt{m}/m = 1/\sqrt{m}$.

Put $A := \{a^{(h)} : h \in \mathcal{H}\} \cup \{-a^{(h)} : h \in \mathcal{H}\}$, a finite set of at most $2\tau_{\mathcal{H}}(2m)$ vectors of Euclidean norm at most $1/\sqrt m$, and note $\sup_h |\langle \sigma, a^{(h)}\rangle| = \max_{a \in A}\langle \sigma, a\rangle$ — this is why the set is symmetrized. Massart's lemma (Lemma 7.3.3, in the un-normalized form $\mathbb{E}_\sigma[\max_{a\in A}\langle\sigma,a\rangle] \leq \max_a\|a\|_2\sqrt{2\log|A|}$) gives

$$\mathbb{E}_\sigma\Big[\max_{a \in A}\langle \sigma, a\rangle\Big] \;\leq\; \frac{1}{\sqrt m}\sqrt{2\log\big(2\tau_{\mathcal{H}}(2m)\big)}.$$

The bound is uniform in $S,S'$, so it survives taking the outer expectation:

$$\mathbb{E}_S\Big[\sup_{h}\big|R(h)-\widehat{R}_S(h)\big|\Big] \;\leq\; \sqrt{\frac{2\log\big(2\tau_{\mathcal{H}}(2m)\big)}{m}} \;=:\; \bar\epsilon_m.$$

*Step 2 (Markov).* The random variable $\sup_h|R(h)-\widehat{R}_S(h)|$ is non-negative, so Theorem 5.6.1 with threshold $\bar\epsilon_m/\delta$ gives $P\big(\sup_h|R(h)-\widehat{R}_S(h)| \geq \bar\epsilon_m/\delta\big) \leq \delta$, which is the claim $\square$

Two remarks on the shape of this bound. First, it depends on $\delta$ as $1/\delta$ rather than $\sqrt{\log(1/\delta)}$, which is the price of using Markov's inequality on the expectation rather than a concentration inequality for the supremum itself; Chapter 7 replaces this step with McDiarmid's inequality and recovers the $\sqrt{\log(1/\delta)}$ dependence. Second, the constants are not optimal — a more careful sub-Gaussian maximal inequality replaces the bound above by $\frac{4+\sqrt{\log \tau_{\mathcal{H}}(2m)}}{\delta\sqrt{2m}}$ — but the essential dependence, $\sqrt{\log\tau_{\mathcal{H}}(2m)/m}$, is the same and is all that the discussion below uses.

This result is consistent with the implications of the fundamental theorem of statistical learning. The bound is useful exactly when $\sqrt{\log \tau_{\mathcal{H}}(2m)}/\sqrt{2m} \rightarrow 0$, i.e. when $\log \tau_{\mathcal{H}}(m)/m \rightarrow 0$: the growth function may grow, but only *sub-exponentially* in $m$. By the Sauer-Shelah-Perles lemma this is precisely what a finite VC dimension guarantees, since $\tau_{\mathcal{H}}(m) \leq (em/d)^d$ is polynomial in $m$, so $\log \tau_{\mathcal{H}}(m) = O(d\log m)$. Conversely, if $\mathcal{H}$ shatters arbitrarily large sets then $\tau_{\mathcal{H}}(m) = 2^m$ for every $m$, $\log \tau_{\mathcal{H}}(m)/m = \log 2$ does not vanish, and the bound degenerates — matching Theorem 6.4.6.

---

## Nonuniform Learnability

**Definition 6.4.10 (Nonuniform Learnability)** A hypothesis class $\mathcal{H}$ is **nonuniformly learnable** if there exist a function $m_{\mathcal{H}}^{NU}: (0, 1)^2 \times \mathcal{H} \to \mathbb{N}$ and a **learning algorithm** $A$ such that for every $\epsilon, \delta \in (0, 1)$ and every distribution $\mathcal{D}$, running $A$ on data set $S=\{(x_i,y_i) \overset{i.i.d.}{\sim} D : i = 1, \ldots, m\}$ satisfies, for every $h \in \mathcal{H}$ and every $m \ge m_{\mathcal{H}}^{NU}(\epsilon, \delta, h)$,

$$P(\{S \sim \mathcal{D}^m: R(A(S)) \le R(h) + \epsilon\}) \geq 1-\delta.$$

The difference from Definition 6.2.5 is that the required sample size may depend on the hypothesis $h$ we are competing against, rather than being uniform over the whole class — hence the name.

---

An agnostic PAC learnable hypothesis class $\mathcal{H}$ is also nonuniform learnable because $R(A(S)) \leq \min_{h' \in \mathcal{H}} R(h') + \epsilon$ results trivially in $R(A(S)) \leq R(h) + \epsilon, \forall h \in \mathcal{H}$. Hence, the set of non-uniform learnable hypothesis classes is larger than the agnostic PAC learnable hypothesis classes. With this relaxation, our motivation is to increase the model capacity while sustaining its learnability.

---

**Theorem 6.4.11** If $\mathcal{H} = \cup_{i \in \mathbb{N}} \mathcal{H}_i$ such that each $\mathcal{H}_i$ has the uniform convergence property, then $\mathcal{H}$ is nonuniformly learnable.

***Proof.*** Fix the weights $w(i) := 6/(\pi^2 i^2)$, which satisfy $\sum_{i\geq1} w(i) = 1$ by the Basel identity $\sum_{i\ge1} i^{-2} = \pi^2/6$, and take $A$ to be the SRM rule $h_{SRM}$ defined below in this section. Theorem 6.4.12, proved next, guarantees that with probability at least $1-\delta$,
$$\begin{gathered}
R(h) \leq \widehat{R}_S(h) + \epsilon_{i(h)}\big(m, w(i(h))\delta\big) \\
\text{simultaneously for every } h \in \mathcal{H}.
\end{gathered}$$
On that event, for any fixed $h \in \mathcal{H}_i$, the definition of $h_{SRM}$ as the minimizer of the right-hand side gives
$$R(h_{SRM}) \leq \widehat{R}_S(h_{SRM}) + \epsilon_{i(h_{SRM})}\big(m,w(i(h_{SRM}))\delta\big) \leq \widehat{R}_S(h) + \epsilon_{i}\big(m,w(i)\delta\big) \leq R(h) + 2\,\epsilon_{i}\big(m,w(i)\delta\big),$$
the last step using the uniform convergence of $\mathcal{H}_i$ once more, in the opposite direction. Since $\mathcal{H}_i$ has the uniform convergence property, $\epsilon_i(m,w(i)\delta) \to 0$ as $m \to \infty$ with $i,\delta$ fixed, so for every $\epsilon,\delta$ and every $h$ there is a finite
$$m_{\mathcal{H}}^{NU}(\epsilon,\delta,h) := \min\big\{ m : 2\,\epsilon_{i(h)}\big(m, w(i(h))\delta\big) \leq \epsilon \big\}$$
beyond which $R(A(S)) \leq R(h)+\epsilon$ holds with probability at least $1-\delta$. This is exactly Definition 6.4.10 $\square$

Note precisely where nonuniformity enters: the sample size above depends on $h$ through the index $i(h)$ of the subclass containing it. There is no single $m$ that works for all $h$ at once, because the penalty $\epsilon_i$ worsens without bound as $i$ grows.

---

Vice versa also holds. If a hypothesis class $\mathcal{H}$ is nonuniformly learnable, then it can be expressed as a countable union of agnostic PAC learnable subclasses. Next let us see how we can use this result to build a more powerful learning algorithm than ERM. Remember that we have sample complexity guarantees at the hypothesis-subclass level while we need to do our search at the hypothesis level. The following quantity is instrumental in bridging this gap: 

$$\epsilon_i(m,\delta) = \min \{\epsilon \in (0,1): m_{\mathcal{H}_i}^{UC}(\epsilon,\delta) \leq m \}$$

which in effect inverts the complexity function $m_{\mathcal{H}_i}^{UC}$ of hypothesis subclass $i$. This quantity gives the smallest representativeness level attainable within subclass $i$ from $m$ data points at confidence $1-\delta$. In other words, the best level of representativeness reachable within this hypothesis subclass, i.e., $P(\{S \sim \mathcal{D}^m: |R(h) - \widehat{R}_S(h)| \leq \epsilon_i(m,\delta)\}) \geq 1-\delta$ for the event of collecting a data set $S$ of size $m$ i.i.d. from any data distribution $\mathcal{D}$.

---

**Theorem 6.4.12** Let $w: \mathbb{N} \rightarrow [0,1]$ be a function with $\sum_{i=1}^\infty w(i) < 1$ and $\mathcal{H} = \cup_{i \in \mathbb{N}} \mathcal{H}_i$ such that each $\mathcal{H}_i$ has the uniform convergence property with $m_{\mathcal{H}_i}^{UC}$. Then for every $\delta \in (0,1)$, every data distribution $\mathcal{D}$, and every $m \in \mathbb{N}$, and every $i \in \mathbb{N}$ and $h \in \mathcal{H}_i$ we have

$$P\left ( \left \{S \sim \mathcal{D}^m: |R(h) - \widehat{R}_S(h)| \leq \epsilon_i(m, w(i) \delta) \right \} \right ) \geq 1-\delta$$

where $S$ is a data set of size $m$ sampled i.i.d. from $\mathcal{D}$.

***Proof.*** Fix $i$ and let $B_i$ be the bad event that $S$ fails to be $\epsilon_i(m, w(i)\delta)$-representative for $\mathcal{H}_i$, i.e. $B_i := \{ \exists h \in \mathcal{H}_i : |R(h)-\widehat{R}_S(h)| > \epsilon_i(m,w(i)\delta)\}$. By the definition of $\epsilon_i$ as the inverse of the sample complexity function $m_{\mathcal{H}_i}^{UC}$, applying Definition 6.2.4 to $\mathcal{H}_i$ with confidence parameter $w(i)\delta$ gives $P(B_i) \leq w(i)\,\delta$. The union bound over the countably many subclasses then gives
$$P\Big(\bigcup_{i \in \mathbb{N}} B_i\Big) \leq \sum_{i \in \mathbb{N}} P(B_i) \leq \delta \sum_{i \in \mathbb{N}} w(i) \;<\; \delta,$$
using the hypothesis $\sum_i w(i) < 1$. On the complement — an event of probability at least $1-\delta$ — no subclass fails, which is the claim $\square$

The proof shows exactly what the weights accomplish: a union bound over infinitely many events is finite only if the individual confidence levels are made summable, and $w$ is the budget allocating a share $w(i)$ of the total failure probability $\delta$ to subclass $i$. Any summable allocation works; different choices of $w$ simply express different prior preferences over the subclasses.

---

The goal of this theorem is to express the uniform convergence of a composite hypothesis class comprising subclasses with different uniform convergence guarantees. It only rescales the confidence level of the subclasses with an index function on the classes. As we will see next, this index will allow us to search the composite hypothesis space and thereby define a learning rule that is more powerful than ERM. The above result can also be expressed as follows:

$$P\left ( \left \{S \sim \mathcal{D}^m: R(h) \leq \widehat{R}_S(h) +  \epsilon_i(m, w(i(h)) \cdot \delta)  \right \} \right ) \geq 1-\delta, \ \forall h \in \mathcal{H}$$

where $i(h)=\min_{i:h \in \mathcal{H}_i}$ is a function that finds the most sample-efficient class a hypothesis $h$ belongs to. This result prescribes a search rule for an extended set of hypotheses that satisfies PAC learnability. We can then use it to build a new learning algorithm

$$h_{SRM} \in \arg \min_{h \in \mathcal{H}} \Big \{ \widehat{R}_S(h) +  \epsilon_{i(h)}(m, w(i(h)) \cdot \delta) \Big \}$$

which we call **Structural Risk Minimization (SRM)**: it minimizes not the empirical risk itself, but the upper bound on $R(h)$ that the previous display certifies. Remember that by Hoeffding's inequality we have $m_h^{UC}(\epsilon,\delta)=\frac{\log(2/\delta)}{2\epsilon^2}$ for a single hypothesis $h$, which implies $\epsilon_i(m,\delta) = \sqrt{\frac{\log(2/\delta)}{2m}}$. Defining $w'(h) := w(i(h))$ we get

$$h_{SRM} \in \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h) +  \sqrt{\frac{-\log w'(h) + \log(2/\delta)}{2m}}$$

where we use the trivial identity that $\log(2/\delta \cdot w'(h)) =-\log w'(h) + \log(2/\delta)$. In effect, SRM applies a preference over the hypothesis subclasses, represented in the learning rule by the weight $w'(h)$. The higher the weight, the stronger the preference. These preferences can be chosen to reflect prior knowledge coming from domain expertise regarding the collected data.

---

In many real-world problems, we have little domain knowledge to incorporate into the learning process. It would be useful to have a guideline for building machine learning algorithms without any domain knowledge. Remember that the only condition for combining individually agnostic PAC learnable hypothesis classes into a bigger class is to define a series of positive weights that converge to a value less than one. The following inequality provides an instrumental way of building such a series.

**Lemma 6.4.13 (Kraft's inequality)** Let $\Sigma^*$ be a set of finite-length strings over the alphabet $\Sigma=\{0,1\}$ that is **prefix-free**, in the sense that no string $s \in \Sigma^*$ of length $k$ coincides with the first $k$ characters of a longer string in $\Sigma^*$. Then $\sum_{s \in \Sigma^*} 2^{-|s|} \leq 1$, where $|s|$ denotes the length of $s$.

***Proof.*** Generate an infinite sequence of i.i.d. fair bits $b_1, b_2, \ldots$, so that $P(b_1 = c_1, \ldots, b_k = c_k) = 2^{-k}$ for any fixed pattern $c \in \{0,1\}^k$. For each $s \in \Sigma^*$ define the event
$$\begin{gathered}
A_s := \{ b_1 = s_1,\ \ldots,\ b_{|s|} = s_{|s|} \}, \\
P(A_s) = 2^{-|s|}.
\end{gathered}$$
The events are **pairwise disjoint**: if $A_s$ and $A_{s'}$ both occurred for $s \neq s'$ with $|s| \leq |s'|$, then $s$ would be the first $|s|$ characters of $s'$, contradicting prefix-freeness. Disjointness and the axioms of a probability measure ($\sigma$-additivity plus $P(\Omega)=1$) give
$$\sum_{s \in \Sigma^*} 2^{-|s|} = \sum_{s \in \Sigma^*} P(A_s) = P\Big(\bigcup_{s \in \Sigma^*} A_s\Big) \leq 1. \qquad\square$$

In words: assigning short codewords is a scarce resource. Prefix-freeness is what makes a code uniquely decodable one symbol at a time, and it forces the codeword lengths to behave like an (unnormalized) probability distribution: one cannot assign a short codeword to every symbol, so some symbols must receive long codewords to pay for the short ones. This is exactly why $2^{-|s(h)|}$ below can be reused as the summable weight $w'(h)$ that Theorem 6.4.12 needs.

The lemma is precisely what Theorem 6.4.12 needs: a description language for hypotheses hands us, for free, a summable weight function $w'(h) := 2^{-|s(h)|}$, with no need to design one by hand.

---

Hence, whenever we can find an alphabet that uniquely describes all hypotheses in a class with a string $s(h)$, we can plug Kraft's inequality into Theorem 6.4.12 with $w'(h) := 2^{-|s(h)|}$ and get

$$P\Big ( \Big \{S \sim \mathcal{D}^m: R(h) \leq \widehat{R}_S(h) +  \sqrt{\frac{|s(h)| + \log(2/\delta)}{2m}} \Big \} \Big ) \geq 1-\delta$$

where we use the property $\log(2) < 1$. The right-hand side of this inequality prescribes the learning rule below
$$h_{MDL} \in \arg \min_{h \in \mathcal{H}} \widehat{R}_S(h) +  \sqrt{\frac{|s(h)| + \log(2/\delta)}{2m}}.$$

Note that this rule suggests the hypothesis that can be expressed with the minimum number of characters among the ones that fit the data best. Hence, it is referred to as the **Minimum Description Length (MDL)** rule. The MDL rule is yet another instance of the Occam's Razor principle in machine learning, which also has overarching consequences on hypothesis development in scientific investigation. This rule also sets a theoretical foundation for the idea of applying **regularizers** in machine learning.

## The Realizable Case: Fast Rates

Theorem 6.1.3 charges a deviation of order $1/\sqrt m$ whatever the risk of the hypothesis. Under realizability (Definition 6.2.1) this is pessimistic: a hypothesis with true risk $\mu$ survives $m$ examples with zero training error with probability $(1-\mu)^m$, which decays exponentially in $m\mu$ rather than in $m\mu^2$. The multiplicative Chernoff bounds of Chapter 5 (Theorem 5.6.10 and Corollary 5.6.11) turn this observation into a $1/m$ rate.

**Theorem 6.4.14 (Realizable case: the fast rate).** Let $\mathcal{H}$ be a finite hypothesis class for binary classification under the zero-one loss, realized by $\mathcal{D}$ (Definition 6.2.1), and let $h_S$ be any ERM hypothesis. Then for any $\delta \in (0,1)$, with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$,
$$R(h_S) \;\leq\; \frac{\log|\mathcal{H}| + \log(1/\delta)}{m}.$$
Moreover — and this second statement survives ERM hypotheses with small *nonzero* training error and arbitrary $\mathcal{D}$ — with probability at least $1-\delta$, simultaneously for all $h \in \mathcal{H}$,
$$R(h) \;\leq\; 2\,\widehat{R}_S(h) + \frac{2\log|\mathcal{H}| + 2\log(1/\delta)}{m}.$$

**Proof.** For the first statement, fix $h \in \mathcal{H}$ with $R(h) = \mu$. Under realizability ERM attains $\widehat{R}_S(h_S) = 0$ (some $h^*$ has zero loss on every sample point, and $h_S$'s training error is minimal). Since the losses $\mathds{1}(h(x_i) \neq y_i)$ are i.i.d. Bernoulli($\mu$),
$$P\big(\widehat{R}_S(h) = 0\big) = (1-\mu)^m \leq e^{-m\mu},$$
the $z=0$ case of Theorem 5.6.10, displayed right after its proof. Union-bounding over the bad events $\{\widehat{R}_S(h) = 0\}$ for those $h$ with $R(h) > \epsilon_0$ (for which $e^{-mR(h)} \leq e^{-m\epsilon_0}$),
$$P\big(\exists h \in \mathcal{H} : \widehat{R}_S(h) = 0,\ R(h) > \epsilon_0\big) \;\leq\; \sum_{h\,:\,R(h)>\epsilon_0} e^{-mR(h)} \;\leq\; |\mathcal{H}|\,e^{-m\epsilon_0}.$$
Setting the right-hand side to $\delta$ gives $\epsilon_0 = \frac{\log|\mathcal{H}|+\log(1/\delta)}{m}$. On the complement of this event, every zero-training-error hypothesis has $R(h) \leq \epsilon_0$ — and $h_S$, having $\widehat{R}_S(h_S)=0$, is one of them.

For the second statement, let $\mu := R(h)$, $x := \widehat{R}_S(h)$, and suppose $\mu > 2x + t$. Then the deviation $\epsilon := \mu - x$ satisfies $\epsilon > x + t \geq t$ and, from $x < (\mu - t)/2$, also $\epsilon > \tfrac{\mu+t}{2}$. Corollary 5.6.11 (lower tail) therefore applies and gives
$$P\big(\mu > 2x + t\big) \;\leq\; P\big(x \leq \mu - \tfrac{\mu+t}{2}\big) \;\leq\; \exp\Big(-\frac{m(\mu+t)^2/4}{2\mu}\Big) \;\leq\; \exp\Big(-\frac{mt}{2}\Big),$$
where the second step uses the deviation $\tfrac{\mu+t}{2} \leq \mu$, i.e. $t \leq \mu$ — and if $t > \mu$ the event on the left is empty, since it forces $x < (\mu-t)/2 < 0$ — and the last step uses the AM-GM inequality $(\mu+t)^2 \geq 4\mu t$. Union-bounding over $|\mathcal{H}|$ fixed hypotheses and setting $|\mathcal{H}|\,e^{-mt/2} = \delta$, i.e. $t = \frac{2\log|\mathcal{H}| + 2\log(1/\delta)}{m}$, gives the simultaneous statement $\square$

The contrast with Theorem 6.1.3 could not be sharper: the deviation scale is now $1/m$, not $1/\sqrt m$. In the first statement the entire mechanism is the multiplicative Chernoff bound — the exponent $m\mu$ is *linear* in the risk being estimated, which is why a hypothesis of true risk, say, $10/m$ has essentially no chance of surviving $m$ examples with zero training error. The second statement is the *relative-deviation* form: the slack $R(h) - 2\widehat{R}_S(h)$ is $O(1/m)$ whenever the empirical risk itself is small, and the first statement is its special case $\widehat{R}_S(h_S) = 0$ (up to the constant $2$). The factor of $2$ in front of $\widehat{R}_S(h)$ is the price of turning a lower-tail bound into a statement uniform over all levels of training error. The same mechanism, run over the finitely many behaviours of an infinite class on a sample, gives the realizable rate of the Fundamental Theorem (Part 2 of the proof of Theorem 6.4.7).

---

Every result in this chapter — the finite-class bound, the VC-dimension-based Fundamental Theorem, the SRM and MDL rules built on top of it — measured the complexity of $\mathcal{H}$ through $|\mathcal{H}|$ or $d_{VC}(\mathcal{H})$: a raw count, or a worst-case combinatorial shattering capacity. Chapter 7 picks up right where the Fundamental Theorem (Theorem 6.4.7) leaves off, and asks what happens if we measure complexity differently — through how well $\mathcal{H}$ can correlate with pure noise (Rademacher complexity), through how finely $\mathcal{H}$ can be approximated by a finite grid (covering numbers), or through how much a data-dependent choice inside $\mathcal{H}$ must diverge from a fixed prior belief (PAC-Bayes bounds) — and shows that all of these, $|\mathcal{H}|$ and $d_{VC}(\mathcal{H})$ included, are instances of one and the same abstract idea.
