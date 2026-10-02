An **ensemble** aggregates the predictions of several hypotheses $h_1,\ldots,h_T$ into a single predictor. The two constructions in this chapter differ in *how* the members are produced and in *which* term of the excess-risk decomposition of Chapter 5 they attack:

| | members produced | aggregation | attacks |
|---|---|---|---|
| **Bagging** | in parallel, on resampled data | unweighted average | the **variance** term |
| **Boosting** | in sequence, on reweighted data | weighted vote $\sum_t w_t h_t$ | the **approximation error** $\epsilon_{app}$ |

Bagging takes a low-bias, high-variance base learner and averages its fluctuations away. Boosting takes a high-bias, low-variance base learner — one barely better than a coin flip — and composes many of them into a predictor of much greater capacity. The second is the more surprising of the two, and it is where the theory of this chapter is concentrated.

# Bootstrap Aggregation (Bagging)

**Bagging** [Breiman (1996)](https://doi.org/10.1007/BF00058655)<!-- cite: breiman1996bagging | article | author={Breiman, Leo}; title={Bagging Predictors}; journal={Machine Learning}; year={1996}; volume={24}; number={2}; pages={123--140} --> is the technique this section formalizes.

**Definition 10.1 (Bootstrap sample).** Given $S = \{(x_i,y_i)\}_{i=1}^m$, a **bootstrap sample** of $S$ is
$$S' := \{(x_{j_k}, y_{j_k}) : j_k \sim \mathcal{U}(\{1,\ldots,m\}),\ k = 1,\ldots,m\},$$
i.e. $m$ indices drawn uniformly **with replacement**, so an index may repeat.

**Definition 10.2 (Bagged predictor).** Let $S_1,\ldots,S_T$ be bootstrap samples of $S$ and $h_{S_t} \gets A(S_t)$. The **bagged predictor** is
$$h_{S_1,\ldots,S_T}(x) := \frac{1}{T}\sum_{t=1}^T h_{S_t}(x).$$

Two elementary facts explain what averaging buys, both stated for the idealized case of $T$ **independent** samples $S_1,\ldots,S_T \sim \mathcal{D}^m$; bootstrapping approximates this by reusing one $S$, at the cost of correlation between the members, which is why the gains below are real but attenuated in practice.

---

**Proposition 10.1 (Averaging preserves bias and divides variance).** Let $h_{S_1},\ldots,h_{S_T}$ be i.i.d. as functions of independent samples $S_1,\ldots,S_T\sim\mathcal{D}^m$. Then for every fixed $x$,
$$\begin{gathered}
\mathbb{E}\Big[\tfrac{1}{T}\textstyle\sum_t h_{S_t}(x)\Big] = \mathbb{E}_S[h_S(x)], \\
\mathrm{Var}\Big[\tfrac{1}{T}\textstyle\sum_t h_{S_t}(x)\Big] = \tfrac{1}{T}\,\mathrm{Var}_S[h_S(x)].
\end{gathered}$$

***Proof.*** The first claim is linearity of expectation. For the second, independence makes all cross-covariances vanish, so
$$\mathrm{Var}\Big[\tfrac1T\sum_t h_{S_t}(x)\Big] = \tfrac{1}{T^2}\sum_{t=1}^T \mathrm{Var}[h_{S_t}(x)] + \tfrac{1}{T^2}\sum_{t \neq t'} \underbrace{\mathrm{Cov}[h_{S_t}(x), h_{S_{t'}}(x)]}_{=\,0} = \tfrac{1}{T^2}\cdot T\,\mathrm{Var}_S[h_S(x)]. \qquad\square$$

In the vocabulary of Chapter 5's bias-variance decomposition: the $\text{Bias}^2$ term is untouched and the $\text{Variance}$ term is divided by $T$. Letting $T\to\infty$, the weak law of large numbers (Theorem 4.3) gives $h_{S_1,\ldots,S_T}(x) \to \bar h(x) := \mathbb{E}_S[h_S(x)]$, the **average predictor** of Chapter 5.

---

**Proposition 10.2 (Bagging never hurts, in expectation).** For the squared loss, $R(\bar h) \leq \mathbb{E}_S[R(h_S)]$, with equality iff $\mathrm{Var}_S[h_S(x)]=0$ for $\mathcal{D}$-almost every $x$.

***Proof.*** Chapter 5 established $\mathbb{E}_S[R(h_S)] - R^*_{\mathcal{D}} = \text{Bias}^2 + \text{Variance}$ and $R(\bar h) - R^*_{\mathcal{D}} = \text{Bias}^2 = \|\bar h - f^*\|^2$. Subtracting, $\mathbb{E}_S[R(h_S)] - R(\bar h) = \text{Variance} = \mathbb{E}_x[\mathrm{Var}_S[h_S(x)]] \geq 0$. Equality holds exactly when the integrand vanishes almost everywhere. (Equivalently: this is Jensen's inequality for the convex map $t \mapsto t^2$.) $\square$

---

Proposition 10.2 compares $\bar h$ only to the *average* member. It can do better. If $\mathcal{H}$ is **not convex**, then $\bar h \notin \mathcal{H}$ in general, and $\bar h$ may be strictly closer to $f^*$ than *every* $h \in \mathcal{H}$ — the caveat already flagged in Chapter 5. Decision trees (Chapter 7) are such a class: an average of trees is not a tree. Bagged trees, with an extra randomization over the features considered at each split, are **random forests**, and this non-convexity is exactly the source of their strength. Bagging a *convex* class such as linear predictors, by contrast, cannot beat its best member, since there $\bar h \in \mathcal{H}$ already.

# Boosting

Boosting starts from the opposite premise: the base class is deliberately **too weak** to fit the data, and the ensemble must manufacture capacity rather than stability. Throughout this section $\mathcal{Y} = \{-1,+1\}$, so that the sign of a real-valued score is a prediction and $y\,f(x) > 0$ means "correct".

## Weak Learnability

**Definition 10.3 ($\gamma$-weak learner).** Let $\gamma \in (0, 1/2)$ and let $\mathcal{H}_0 \subseteq \{-1,+1\}^{\mathcal{X}}$ be a **base class**. A **$\gamma$-weak learner** for $\mathcal{H}_0$ is a procedure $\mathrm{WL}$ that maps a sample $S$ and a probability vector $D \in \Delta_m := \{D \in \mathbb{R}^m_{\geq0} : \sum_i D_i = 1\}$ to some $h = \mathrm{WL}(S,D) \in \mathcal{H}_0$ whose **$D$-weighted empirical error** satisfies
$$\widehat{R}_{S,D}(h) := \sum_{i=1}^m D_i\, \mathds{1}\big(h(x_i) \neq y_i\big) \;\leq\; \tfrac12 - \gamma .$$

---

The demand is minimal: a coin flip achieves $1/2$ in expectation, so $\mathrm{WL}$ must merely beat guessing by a fixed margin $\gamma$ — but it must do so for **every** reweighting $D$ of the sample, including adversarial ones that concentrate on the points previous rounds got wrong. This uniformity over $D$ is the whole content of the assumption, and it is what boosting consumes.

Definition 10.3 is the *empirical* form of weak learnability, stated over reweightings of a fixed sample because that is exactly the interface AdaBoost calls. Its distributional counterpart replaces "$D$ on $[m]$" by "any distribution $\mathcal{D}$ over $\mathcal{X}\times\mathcal{Y}$ realizable by $\mathcal{H}_0$" and asks for $R(h) \leq \tfrac12-\gamma$ with high probability from finitely many samples. The two are linked by the Fundamental Theorem (Theorem 5.4): if $d_{VC}(\mathcal{H}_0) < \infty$ then ERM over $\mathcal{H}_0$ is a $\gamma$-weak learner in the distributional sense for every $\gamma < 1/2$, given enough samples — so weak learnability is not a weaker *combinatorial* condition than PAC learnability, only a weaker *accuracy* requirement. Corollary 10.1 below shows that even that weakening buys nothing: the two are equivalent.

**Definition 10.4 (Class of $T$-term linear combinations).** For a base class $\mathcal{H}_0$,
$$L(\mathcal{H}_0, T) := \Big\{ x \mapsto \mathrm{sign}\Big(\sum_{t=1}^T w_t h_t(x)\Big) \;:\; w \in \mathbb{R}^T,\ h_1,\ldots,h_T \in \mathcal{H}_0 \Big\}.$$

$\mathcal{H}_0 \subseteq L(\mathcal{H}_0,1) \subseteq L(\mathcal{H}_0,2) \subseteq \cdots$, so the ensemble class is a **nested family** indexed by $T$ — precisely the setting of nonuniform learnability and SRM (Chapter 5), with the number of rounds $T$ playing the role that the leaf budget played for decision trees in Chapter 7.

**Example (decision stumps).** $\mathcal{H}_{\mathrm{stump}} := \{x \mapsto s\,\mathrm{sign}(x_j - \theta) : j \in [d],\ \theta \in \mathbb{R},\ s \in \{-1,+1\}\}$. On any $m$ points, a stump's labeling is determined by the feature $j$ ($d$ choices), the position of $\theta$ among the $m$ observed values of that feature ($m+1$ choices), and the polarity $s$ ($2$ choices), so
$$\tau_{\mathcal{H}_{\mathrm{stump}}}(m) \leq 2d(m+1).$$
Since $\tau(m) = 2^m$ is needed to shatter, $d_{VC}(\mathcal{H}_{\mathrm{stump}}) = O(\log d)$: a stump is a genuinely weak hypothesis, unable even to represent a two-feature XOR. Yet $L(\mathcal{H}_{\mathrm{stump}}, T)$ grows rich with $T$, which is what makes stumps the canonical base class for boosting.

## AdaBoost

**AdaBoost** [Freund and Schapire (1997)](https://doi.org/10.1006/jcss.1997.1504)<!-- cite: freund1997decisiontheoretic | article | author={Freund, Yoav and Schapire, Robert E.}; title={A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting}; journal={Journal of Computer and System Sciences}; year={1997}; volume={55}; number={1}; pages={119--139} --> is the algorithm below.

**Input:** $S = \{(x_i,y_i)\}_{i=1}^m$ with $y_i \in \{-1,+1\}$; a weak learner $\mathrm{WL}$; a number of rounds $T$.

1. $D^{(1)} := \big(\tfrac1m, \ldots, \tfrac1m\big)$.
2. For $t = 1,\ldots,T$:
    1. $h_t := \mathrm{WL}\big(S, D^{(t)}\big)$
    2. $\displaystyle \epsilon_t := \sum_{i=1}^m D_i^{(t)}\,\mathds{1}\big(h_t(x_i) \neq y_i\big)$
    3. $\displaystyle w_t := \tfrac{1}{2}\log\Big(\frac{1-\epsilon_t}{\epsilon_t}\Big)$
    4. $\displaystyle D_i^{(t+1)} := \frac{D_i^{(t)}\, e^{-w_t y_i h_t(x_i)}}{\sum_{j=1}^m D_j^{(t)}\, e^{-w_t y_j h_t(x_j)}}, \qquad i = 1,\ldots,m$
3. **Output** $h_S := \mathrm{sign}(f_T) \in L(\mathcal{H}_0,T)$, where $\displaystyle f_T(x) := \sum_{t=1}^T w_t\, h_t(x)$.

Step 2.4 is one line because $y_i h_t(x_i) = +1$ on correct points and $-1$ on incorrect ones: the update multiplies a correct point's weight by $e^{-w_t}$ and an incorrect point's by $e^{+w_t}$, then renormalizes. The choice of $w_t$ in step 2.3 is not a guess; it is a minimization, as the proof below shows in one line.

---

**Theorem 10.1 (AdaBoost drives the training error down exponentially).** If $\epsilon_t \leq \tfrac12 - \gamma$ at every round, then the output of AdaBoost satisfies
$$\widehat{R}_S(h_S) = \frac{1}{m}\sum_{i=1}^m \mathds{1}\big(h_S(x_i) \neq y_i\big) \;\leq\; e^{-2\gamma^2 T}.$$

***Proof.*** Write $f_t := \sum_{p \leq t} w_p h_p$, $f_0 := 0$, and define the **exponential loss**
$$Z_t := \frac{1}{m}\sum_{i=1}^m e^{-y_i f_t(x_i)}.$$

*Step 1: the exponential loss dominates the zero-one loss.* Since $\mathds{1}(u \leq 0) \leq e^{-u}$ for all $u \in \mathbb{R}$, and $h_S(x_i)\neq y_i$ exactly when $y_i f_T(x_i) \leq 0$,
$$\widehat{R}_S(h_S) \leq Z_T .$$

*Step 2: the weights are the normalized exponential losses.* We claim
$$D_i^{(t)} = e^{-y_i f_{t-1}(x_i)} / \sum_j e^{-y_j f_{t-1}(x_j)}.$$
For $t=1$ both sides are $1/m$ since $f_0 = 0$. Assuming it at $t$, step 2.4 gives
$$D_i^{(t+1)} \;\propto\; D_i^{(t)} e^{-w_t y_i h_t(x_i)} \;\propto\; e^{-y_i f_{t-1}(x_i)} e^{-w_t y_i h_t(x_i)} = e^{-y_i f_t(x_i)},$$
and normalizing proves the claim at $t+1$.

*Step 3: the per-round progress factor.* Using Step 2, and splitting the sum according to whether $h_t$ is correct,
$$
\begin{aligned}
\frac{Z_t}{Z_{t-1}} = \frac{\sum_i e^{-y_i f_{t-1}(x_i)}\,e^{-w_t y_i h_t(x_i)}}{\sum_i e^{-y_i f_{t-1}(x_i)}} = \sum_{i=1}^m D_i^{(t)} e^{-w_t y_i h_t(x_i)} = \underbrace{(1-\epsilon_t)\,e^{-w_t} + \epsilon_t\, e^{w_t}}_{=:\ \psi(w_t)} .
\end{aligned}
$$
Now $\psi'(w) = -(1-\epsilon_t)e^{-w} + \epsilon_t e^{w} = 0$ gives $e^{2w} = (1-\epsilon_t)/\epsilon_t$, i.e. **exactly the $w_t$ of step 2.3** — this is where that formula comes from, and $\psi$ is convex, so it is a minimum. Substituting $e^{w_t} = \sqrt{(1-\epsilon_t)/\epsilon_t}$,
$$\frac{Z_t}{Z_{t-1}} = (1-\epsilon_t)\sqrt{\tfrac{\epsilon_t}{1-\epsilon_t}} + \epsilon_t\sqrt{\tfrac{1-\epsilon_t}{\epsilon_t}} = 2\sqrt{\epsilon_t(1-\epsilon_t)}.$$

*Step 4: telescoping.* Writing $\epsilon_t = \tfrac12 - \gamma_t$ with $\gamma_t \geq \gamma$, we get $2\sqrt{\epsilon_t(1-\epsilon_t)} = \sqrt{1-4\gamma_t^2} \leq e^{-2\gamma_t^2} \leq e^{-2\gamma^2}$, using $1-u \leq e^{-u}$ with $u = 4\gamma_t^2$ under the square root. Since $Z_0 = 1$,
$$\widehat{R}_S(h_S) \leq Z_T = \prod_{t=1}^T \frac{Z_t}{Z_{t-1}} \leq e^{-2\gamma^2 T}. \qquad\square$$

Two readings of the proof are worth keeping. First, AdaBoost is **coordinate-wise greedy minimization of the exponential loss** $Z_t$: step 2.1 picks the direction $h_t$ and step 2.3 picks the optimal step size $w_t$ along it. Second, the weighting scheme is not an extra heuristic layered on top — Step 2 shows $D^{(t)}$ *is* the exponential loss, normalized, so up-weighting the currently misclassified points is exactly what greedy minimization of $Z_t$ prescribes.

---

**Corollary 10.1 (Weak learnability implies strong learnability).** Let $\mathcal{H}_0$ admit a $\gamma$-weak learner and run AdaBoost for $T \geq \frac{\log m}{2\gamma^2}$ rounds. Then $\widehat{R}_S(h_S) = 0$.

***Proof.*** By Theorem 10.1, $\widehat{R}_S(h_S) \leq e^{-2\gamma^2 T} \leq e^{-\log m} = 1/m$. But $\widehat{R}_S(h_S)$ is a multiple of $1/m$, and the only such value strictly below $1/m$ is $0$. $\square$

This is the theorem that made boosting famous: a learner that can only beat a coin flip by $\gamma$, on every reweighting of the sample, can be compiled into one that fits the sample perfectly, in $O(\gamma^{-2}\log m)$ rounds. **Weak** and **strong** learnability are the same notion, up to computation. What remains is to price the generalization of the resulting predictor — and this is where two different complexity measures give two genuinely different answers.

## What Linear Combinations Cost: Two Accounts

### Account 1: the VC dimension of $L(\mathcal{H}_0,T)$ grows with $T$

**Lemma 10.1.** Let $d_0 := d_{VC}(\mathcal{H}_0) < \infty$ (not to be confused with the feature dimension $d$). Then
$$
\begin{aligned}
\tau_{L(\mathcal{H}_0,T)}(m) &\;\leq\; \tau_{\mathcal{H}_0}(m)^T \cdot \Big(\frac{em}{T}\Big)^{T} \;\leq\; \Big(\frac{em}{d_0}\Big)^{d_0T}\Big(\frac{em}{T}\Big)^{T}, \\
&\qquad\text{hence}\qquad d_{VC}\big(L(\mathcal{H}_0,T)\big) = O\big(d_0T\log(d_0T)\big).
\end{aligned}
$$

***Proof.*** Fix $S = \{x_1,\ldots,x_m\}$. A hypothesis of $L(\mathcal{H}_0,T)$ is specified by base hypotheses $h_1,\ldots,h_T$ and weights $w$. Its restriction to $S$ depends on the $h_t$ only through their own restrictions to $S$, and there are at most $\tau_{\mathcal{H}_0}(m)$ of those per slot, hence at most $\tau_{\mathcal{H}_0}(m)^T$ jointly. Having fixed them, map each point to $v_i := (h_1(x_i),\ldots,h_T(x_i)) \in \{-1,+1\}^T$; the remaining freedom is the labeling $\mathrm{sign}(w^\top v_i)$, i.e. a homogeneous halfspace in $\mathbb{R}^T$ applied to $m$ fixed points. The class of homogeneous halfspaces in $\mathbb{R}^T$ has VC dimension $T$, so by Sauer-Shelah-Perles (Lemma 5.2) it realizes at most $(em/T)^T$ labelings. Multiplying the two counts gives the first claim, and the second follows from Lemma 5.2 applied to $\mathcal{H}_0$. For the VC bound, shattering $m$ points requires $2^m \leq \tau_{L(\mathcal{H}_0,T)}(m)$, i.e. $m\log 2 \leq (d_0T+T)\log(em)$, which forces $m = O(d_0T\log(d_0T))$. $\square$

Feeding $d_{VC}(L(\mathcal{H}_0,T)) = O(d_0T\log(d_0T))$ into the Fundamental Theorem (Theorem 5.4) gives, with probability at least $1-\delta$, for every $h \in L(\mathcal{H}_0,T)$,

$$R(h) \;\leq\; \widehat{R}_S(h) + O\left(\sqrt{\frac{d_0T\log(d_0T) + \log(1/\delta)}{m}}\right). \tag{10.1}$$

Combined with Corollary 10.1 this already proves boosting works: choose $T \asymp \gamma^{-2}\log m$, get $\widehat{R}_S = 0$, and read off a nonvacuous bound as soon as $m \gtrsim d_0\gamma^{-2}\log^2 m$. It also makes a **prediction**: the bound degrades as $\sqrt{T}$, so running AdaBoost past the point where it fits the sample should start to overfit. Empirically, this prediction fails — AdaBoost's test error typically keeps *improving* for hundreds of rounds after the training error hits zero. The bound is not wrong; it is measuring the wrong thing.

### Account 2: the Rademacher complexity of the convex hull does not grow at all

Assume $\mathcal{H}_0$ is **symmetric** ($h \in \mathcal{H}_0 \Rightarrow -h \in \mathcal{H}_0$; true for stumps, by flipping $s$). Then we may take every $w_t \geq 0$ without loss of generality, and the **normalized ensemble**
$$\begin{gathered}
\bar f_T(x) := \frac{f_T(x)}{\sum_{t=1}^T w_t} = \sum_{t=1}^T \lambda_t h_t(x), \\
\lambda_t := \frac{w_t}{\sum_p w_p} \geq 0,\ \ \sum_t \lambda_t = 1,
\end{gathered}$$
is a **convex combination** of members of $\mathcal{H}_0$, i.e. $\bar f_T \in \mathrm{conv}(\mathcal{H}_0)$, with $\bar f_T(x) \in [-1,1]$. Note $\mathrm{sign}(\bar f_T) = \mathrm{sign}(f_T) = h_S$: normalizing changes no prediction, only the scale on which we may measure *how confidently* each prediction was made.

---

**Theorem 10.2 (Convexification is free).** For any $A \subset \mathbb{R}^m$, $\widehat{\mathfrak{R}}(\mathrm{conv}(A)) = \widehat{\mathfrak{R}}(A)$. In particular $\widehat{\mathfrak{R}}_S(\mathrm{conv}(\mathcal{H}_0)) = \widehat{\mathfrak{R}}_S(\mathcal{H}_0)$ for every $S$, **independently of $T$**.

***Proof.*** $A \subseteq \mathrm{conv}(A)$ gives "$\geq$". For "$\leq$", fix the signs $\sigma \in \{-1,+1\}^m$ and let $a = \sum_j \lambda_j a^{(j)} \in \mathrm{conv}(A)$ with $\lambda_j \geq 0$, $\sum_j \lambda_j = 1$. By linearity of the inner product,
$$\langle \sigma, a\rangle = \sum_j \lambda_j \langle \sigma, a^{(j)}\rangle \leq \Big(\sum_j \lambda_j\Big) \max_j \langle \sigma, a^{(j)}\rangle \leq \sup_{a' \in A} \langle \sigma, a'\rangle .$$
Taking the supremum over $a \in \mathrm{conv}(A)$ and then $\mathbb{E}_\sigma$ of both sides gives the claim. $\square$

---

**Theorem 10.3 (Margin bound for boosting).** Let $\mathcal{H}_0$ be symmetric with values in $\{-1,+1\}$, fix a **margin threshold** $\nu \in (0,1]$ — a free parameter of the bound, unrelated to the weak-learning parameter $\gamma$ of Definition 10.3 — and let $\bar f \in \mathrm{conv}(\mathcal{H}_0)$ be arbitrary. Then with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$, simultaneously for every such $\bar f$,
$$
\begin{aligned}
R\big(\mathrm{sign}(\bar f)\big) &\;=\; P_{(x,y)\sim\mathcal{D}}\big(y\,\bar f(x) \leq 0\big) \;\leq\; \underbrace{\frac{1}{m}\sum_{i=1}^m \mathds{1}\big(y_i \bar f(x_i) \leq \nu\big)}_{\text{fraction of training margins} \,\leq\, \nu} \\
&\;+\; \frac{2}{\nu}\,\widehat{\mathfrak{R}}_S(\mathcal{H}_0) \;+\; 3\sqrt{\frac{\log(2/\delta)}{2m}} .
\end{aligned}
$$

***Proof.*** Call $y\bar f(x) \in [-1,1]$ the **margin** of the prediction — positive exactly when the weighted vote is correct, and large when it is correct by a wide majority. Define the **$\nu$-margin loss**
$$\phi_\nu(u) := \min\big(1, \max(0, 1 - u/\nu)\big),$$
which is $\tfrac{1}{\nu}$-Lipschitz, takes values in $[0,1]$, and satisfies the sandwich $\mathds{1}(u \leq 0) \leq \phi_\nu(u) \leq \mathds{1}(u \leq \nu)$.

Apply the Rademacher generalization bound (Theorem 5a.2) to the loss class $\big\{(x,y)\mapsto \phi_\nu\big(y\bar f(x)\big) : \bar f \in \mathrm{conv}(\mathcal{H}_0)\big\}$, whose values lie in $[0,1]$: with probability $\geq 1-\delta$, for all $\bar f$,
$$\mathbb{E}_{\mathcal{D}}\big[\phi_\nu(y\bar f(x))\big] \leq \frac1m\sum_i \phi_\nu\big(y_i \bar f(x_i)\big) + 2\,\widehat{\mathfrak{R}}_S\big(\phi_\nu \circ \mathrm{conv}(\mathcal{H}_0)\big) + 3\sqrt{\tfrac{\log(2/\delta)}{2m}} .$$
The left side is $\geq P_{\mathcal{D}}\big(y\bar f(x)\leq 0\big)$ and the empirical average is $\leq \frac1m\sum_i \mathds{1}(y_i\bar f(x_i)\leq\nu)$, both by the sandwich. For the middle term, note that the contraction lemma (Lemma 5a.2) permits a *different* function in each coordinate: take $\phi_i(u) := \phi_\nu(y_i u)$, which is $\tfrac1\nu$-Lipschitz for either sign of $y_i$, so contracting with these removes both the label multiplication and the margin loss in one step. Theorem 10.2 then removes the convex hull:
$$\widehat{\mathfrak{R}}_S\big(\phi_\nu \circ \mathrm{conv}(\mathcal{H}_0)\big) \;\leq\; \tfrac1\nu\,\widehat{\mathfrak{R}}_S\big(\mathrm{conv}(\mathcal{H}_0)\big) \;=\; \tfrac1\nu\,\widehat{\mathfrak{R}}_S(\mathcal{H}_0). \qquad\square$$

---

**The resolution.** Compare (10.1) with Theorem 10.3, term by term:

| | Account 1 (VC of $L(\mathcal{H}_0,T)$) | Account 2 (Rademacher of $\mathrm{conv}(\mathcal{H}_0)$) |
|---|---|---|
| fit term | $\widehat{R}_S(h_S)$, hits $0$ and then stops moving | $\frac1m\sum_i\mathds{1}(y_i\bar f_T(x_i)\leq\nu)$, **keeps shrinking** |
| complexity term | $O(\sqrt{d_0T\log(d_0T)/m})$, **grows with $T$** | $\frac{2}{\nu}\widehat{\mathfrak{R}}_S(\mathcal{H}_0)$, **independent of $T$** |

Account 2 says that additional boosting rounds cost nothing in complexity — a convex combination of $T$ members is no richer, in the Rademacher sense, than a single member — while they continue to buy something the zero-one training error is too coarse to see: Theorem 10.1's proof shows AdaBoost keeps driving the exponential loss $Z_t$ down after $\widehat{R}_S = 0$, and $Z_t$ is small only when the margins $y_i f_t(x_i)$ are *large*, not merely positive. Boosting past zero training error is therefore not idle: it is converting correct-but-marginal predictions into correct-and-confident ones, which is precisely what the first term of Theorem 10.3 rewards. This is the same lesson as Chapter 8's: the complexity measure that grows with the number of parameters is the wrong one, and a scale-sensitive measure — margins here, weight norms there — is the right one.

The trade-off has not disappeared, it has moved into $\nu$: a larger $\nu$ shrinks the complexity term but makes the margin-violation term larger. The bound holds for all $\nu$ simultaneously (up to a union bound over a grid of $\nu$'s), so one may read it at whichever $\nu$ is most favorable for the ensemble actually obtained.

## Implementation: AdaBoost with Decision Stumps

We implement Definition 10.3 and the AdaBoost pseudocode literally: the weak learner is an exhaustive search over $\mathcal{H}_{\mathrm{stump}}$ for the stump of least $D$-weighted error, and the boosting loop is the four numbered steps. The data set is the **Wisconsin Breast Cancer** set of Chapters 6 and 7 ($569$ samples, $d=30$ features), with labels recoded to $\{-1,+1\}$. Alongside the two error curves we track $\min_i y_i \bar f_T(x_i)$, the smallest training margin — the quantity Theorem 10.3 says should keep improving after the training error stops.

```python
# AdaBoost implemented from scratch: the base class is decision stumps, the
# weak learner is an exhaustive weighted-error search over them, and the
# boosting loop is line-for-line the pseudocode above.

import torch as th
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

th.manual_seed(0)

data = load_breast_cancer()
X_all = th.tensor(data.data, dtype=th.float64)
y_all = 2 * th.tensor(data.target, dtype=th.float64) - 1   # labels in {-1,+1}
idx = th.randperm(y_all.numel())
n_tr = int(0.7 * y_all.numel())
tr, te = idx[:n_tr], idx[n_tr:]
X, y = X_all[tr], y_all[tr]
X_te, y_te = X_all[te], y_all[te]
m, d = X.shape
order = th.argsort(X, dim=0)                           # sort once, reuse

# --- The base class: decision stumps  h(x) = s * sign(x_j - theta) ----------

def weak_learner(D):
    """Return the stump of least D-weighted empirical error, by exhaustive
    search over all d features and all m+1 distinct thresholds of each."""
    best = (float('inf'), 0, -float('inf'), 1)         # (err, j, theta, s)
    for j in range(d):
        o = order[:, j]
        xs, ys, Ds = X[o, j], y[o], D[o]
        # Sweep theta from left to right. At theta = -inf the stump predicts
        # +1 everywhere, so its error is the weight of the negative points;
        # each time theta passes a point, that point's prediction flips,
        # changing the error by +D_i if y_i = +1 and by -D_i if y_i = -1.
        errs = Ds[ys == -1].sum() + th.cat([th.zeros(1, dtype=D.dtype),
                                            th.cumsum(Ds * ys, dim=0)])
        thetas = th.cat([th.tensor([-float('inf')], dtype=xs.dtype), xs])
        for s in (1, -1):                              # both leaf polarities
            e = errs if s == 1 else 1.0 - errs
            k = int(e.argmin())
            if e[k] < best[0]:
                best = (e[k].item(), j, thetas[k].item(), s)
    return best[1:]                                    # (j, theta, s)

def stump_predict(Xq, stump):
    j, theta, s = stump
    return s * th.where(Xq[:, j] > theta, 1.0, -1.0).double()

# --- AdaBoost ---------------------------------------------------------------

T = 200
D = th.full((m,), 1.0 / m, dtype=th.float64)           # step 1: D^(1)
w_sum = 0.0
F, F_te = th.zeros(m, dtype=th.float64), th.zeros(y_te.numel(), dtype=th.float64)
err_tr, err_te, min_margin = [], [], []

for t in range(T):
    stump = weak_learner(D)                            # 2.1  h_t = WL(S,D^(t))
    pred = stump_predict(X, stump)
    eps = D[pred != y].sum()                           # 2.2  epsilon_t
    eps = eps.clamp(1e-10, 1 - 1e-10)                  #      guard log(0)
    w_t = 0.5 * th.log(1.0 / eps - 1.0)                # 2.3  w_t
    D = D * th.exp(-w_t * y * pred)                    # 2.4  reweight ...
    D = D / D.sum()                                    #      ... and normalise

    w_sum += w_t.item()                                # bookkeeping only:
    F += w_t * pred                                    #   f_t on the train set
    F_te += w_t * stump_predict(X_te, stump)           #   f_t on the test set
    err_tr.append((th.sign(F) != y).double().mean().item())
    err_te.append((th.sign(F_te) != y_te).double().mean().item())
    min_margin.append((y * F / w_sum).min().item())    # min_i y_i bar-f_t(x_i)

t0 = next(t for t, e in enumerate(err_tr) if e == 0.0)
print("       T   train err   test err   min margin")
for t in [0, 4, 9, t0, 49, 99, T - 1]:
    print(f"{t+1:8d}    {err_tr[t]:.4f}     {err_te[t]:.4f}     {min_margin[t]:+.4f}")
print(f"\ntraining error first reaches 0 at T = {t0+1}")

rounds = th.arange(1, T + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(rounds, err_tr, label=r"training error $\widehat{R}_S(h_S)$ (S = training set)")
ax[0].plot(rounds, err_te, label=r"test error (estimate of $R(h_S)$)")
ax[0].axvline(t0 + 1, ls=":", c="k")
ax[0].set_xscale("log"); ax[0].set_xlabel("boosting rounds $T$")
ax[0].set_ylabel("zero-one error"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(rounds, min_margin, c="C2")
ax[1].axvline(t0 + 1, ls=":", c="k"); ax[1].axhline(0, lw=.5, c="k")
ax[1].set_xscale("log"); ax[1].set_xlabel("boosting rounds $T$")
ax[1].set_ylabel(r"minimum training margin")
ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
```

           T   train err   test err   min margin
           1    0.0804     0.0819     -1.0000
           5    0.0251     0.0585     -0.6193
          10    0.0201     0.0585     -0.1861
          27    0.0000     0.0292     +0.0253
          50    0.0000     0.0351     +0.0599
         100    0.0000     0.0292     +0.1175
         200    0.0000     0.0234     +0.1268

    training error first reaches 0 at T = 27

![AdaBoost's training and test error (left) and minimum training margin (right) as a function of the number of boosting rounds $T$, on a logarithmic axis; the dotted line marks the round at which training error first reaches zero.](fig/generated/10_Ensemble_Methods_1.png)

The numbers track the theory. Across the $200$ rounds the worst weighted error the stump search ever returns is $\epsilon_t = 0.436$, so the weak-learning parameter realized on this data set is $\gamma = 0.065$. Theorem 10.1 then guarantees only $\widehat{R}_S \leq e^{-2(0.065)^2\cdot 27} \approx 0.80$ after $27$ rounds, and Corollary 10.1's sufficient budget is $\log(398)/(2\cdot0.065^2) \approx 720$ rounds — both valid but, as worst-case bounds usually are, far more conservative than what actually happens: the training error is already exactly $0$ at $T = 27$. From $T=27$ onward, the first term of Account 1 is frozen at $0$ and its complexity term keeps growing, so (10.1) predicts deterioration; instead the test error *falls further*, from $2.9\%$ to $2.3\%$. Account 2 explains why: over the same rounds the smallest training margin climbs from $+0.025$ to $+0.127$, so the margin-violation term of Theorem 10.3 keeps shrinking at any fixed threshold $\nu \lesssim 0.12$, while its complexity term $\frac2\nu \widehat{\mathfrak{R}}_S(\mathcal{H}_{\mathrm{stump}})$ never moved.
