The exam contains 13 questions. All questions ask for an evaluation of five statements with (yes/no) or (true/false) answers. A true answer adds $+1$ points and a false answer adds $-1$ points to your exam score. Hence, skipping to evaluate a statement is very likely to be more advantageous for you than making a random guess. Your final score is calculated as (Your point sum)$/(13 \times 5)\times 100$. The exam duration is four hours.

**Notation.** The conventions are those of the lecture notes. $P(\cdot)$ denotes a probability — of an event, or of a discrete random variable taking a value; lowercase $p(\cdot)$ is a density. $\mathbb{E}[\cdot]$, $\mathrm{Var}[\cdot]$, $\mathrm{Cov}[\cdot,\cdot]$ carry subscripts naming the variable averaged over; $\mathbb{1}(\cdot)$ is the indicator and $\log$ the natural logarithm. $S=\{(x_i,y_i)\}_{i=1}^m \overset{\text{i.i.d.}}{\sim}\mathcal{D}$ over $\mathcal{X}\times\mathcal{Y}$, with $\mathcal{D}_{\mathcal{X}}$ the marginal on $\mathcal{X}$; $d$ is the feature dimension, $C$ the number of classes, $[m]:=\{1,\ldots,m\}$. $\ell(y,\widehat y)$ is the pointwise loss (true label first), $R(h)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(y,h(x))]$, $\widehat R_S(h)=\frac1m\sum_i \ell(y_i,h(x_i))$, $R^*_{\mathcal{D}}$ the Bayes error, $\tau_{\mathcal{H}}$ the growth function, $d_{VC}$ the VC dimension, $\widehat{\mathfrak{R}}$ the empirical Rademacher complexity. $\top$ is transpose.

**Every statement below is a mathematical assertion about the setup stated in its question**, decidable by proof or by counterexample from the results of the course. No numerical evaluation is required anywhere. **No assumptions may be made additional to those specified in the question text.**

---

## Question 1

Let $\mathcal{X}=[0,1]$, $\mathcal{Y}=\{0,1\}$, and let $\ell$ be the zero-one loss. Let $\mathcal{D}$ be such that $\mathcal{D}_{\mathcal{X}}$ is the uniform distribution on $[0,1]$ and $y=f(x)$ for a fixed, unknown $f:\mathcal{X}\to\{0,1\}$. Write $q:=P_{x\sim\mathcal{D}_{\mathcal{X}}}\big(f(x)=1\big)$ and let $\mathcal{H}_{\mathrm{all}} := \{0,1\}^{\mathcal{X}}$ be the class of **all** functions $\mathcal{X}\to\{0,1\}$. Given $S$, define

$$h_S(x) := \begin{cases} y_i, & \text{if } \exists i \in [m] \text{ with } x_i = x,\\ 0, & \text{otherwise.}\end{cases}$$

Evaluate the following statements. All probabilistic claims are over the draw of $S\sim\mathcal{D}^m$.

**Answer 1.1:** $\widehat R_S(h_S)=0$ with probability $1$.

**Answer 1.2:** $R(h_S)=q$ with probability $1$.

**Answer 1.3:** $h_S \in \arg\min_{h\in\mathcal{H}_{\mathrm{all}}}\widehat R_S(h)$ with probability $1$, but this $\arg\min$ is not a singleton.

**Answer 1.4:** For every $f$, $\ \sup_{h\in\mathcal{H}_{\mathrm{all}}}\big(R(h)-\widehat R_S(h)\big)=1$ with probability $1$.

**Answer 1.5:** Because $h_S$ is an empirical risk minimizer over $\mathcal{H}_{\mathrm{all}}$ and $\min_{h\in\mathcal{H}_{\mathrm{all}}}R(h)=0$, Lemma 5.1 implies that for every $\epsilon>0$ there is an $m$ beyond which $R(h_S)\le\epsilon$.

---

## Question 2

Let $Z_1,\ldots,Z_m$ be i.i.d. with $P(Z_i\in[0,1])=1$, $\mathbb{E}[Z_i]=\mu$, $\mathrm{Var}[Z_i]=\sigma^2>0$, and $\widehat\mu_m := \frac1m\sum_{i=1}^m Z_i$. Fix $\delta\in(0,1)$ and define

$$B_{\mathrm{Cheb}}(\delta,m) := \frac{\sigma}{\sqrt{m\delta}}, \qquad B_{\mathrm{Hoef}}(\delta,m) := \sqrt{\frac{\log(1/\delta)}{2m}}.$$

Separately, let $\mathcal{H}$ be a hypothesis class, $\ell$ take values in $[0,1]$, and $\Phi(S) := \sup_{h\in\mathcal{H}}\big(R(h)-\widehat R_S(h)\big)$. Evaluate the following statements.

**Answer 2.1:** $P\big(|\widehat\mu_m-\mu| \ge B_{\mathrm{Cheb}}(\delta,m)\big)\le\delta$ for every $m\ge1$.

**Answer 2.2:** Establishing $P\big(\widehat\mu_m-\mu \ge B_{\mathrm{Hoef}}(\delta,m)\big)\le\delta$ requires knowledge of $\sigma^2$.

**Answer 2.3:** The ratio $B_{\mathrm{Cheb}}(\delta,m)/B_{\mathrm{Hoef}}(\delta,m)$ is independent of $m$.

**Answer 2.4:** For a finite $\mathcal{H}$, the derivation of Theorem 5.1 requires the random variables $\ell(y_i,h(x_i))$ and $\ell(y_i,h'(x_i))$ to be independent for $h\ne h'$.

**Answer 2.5:** $\Phi$ satisfies the bounded-differences condition of Theorem 5a.1 with $c_i=1/m$ for every $i$, whether $\mathcal{H}$ is finite or infinite.

---

## Question 3

**Model A.** $\theta\sim\mathrm{Beta}(\alpha,\beta)$ with $\alpha,\beta>0$, and $x_1,\ldots,x_m\mid\theta \overset{\text{i.i.d.}}{\sim}\mathrm{Bernoulli}(\theta)$. Write $k:=\sum_{i=1}^m x_i$. It is given that $p(\theta\mid S)=\mathrm{Beta}(\theta\mid \alpha+k,\ \beta+m-k)$, that $\theta_{MLE}=k/m$, that $\theta_{MAP}=\frac{\alpha+k-1}{\alpha+\beta+m-2}$ when $\alpha+k>1$ and $\beta+m-k>1$, and that $P(x_*{=}1\mid S)=\frac{\alpha+k}{\alpha+\beta+m}$.

**Model B.** $y\in[C]$ and $x\in\{0,1\}^d$, with $P(y{=}c)=\pi_c$ and, under the naive Bayes assumption, $P(x\mid y{=}c)=\prod_{j=1}^d \theta_{j,c}^{x_j}(1-\theta_{j,c})^{1-x_j}$.

Evaluate the following statements.

**Answer 3.1:** In Model A with $\alpha=\beta=1$, $\ \theta_{MAP}=\theta_{MLE}$ for every $m\ge2$ and every $S$ with $0<k<m$.

**Answer 3.2:** In Model A, $P(x_*{=}1\mid S)=\theta_{MAP}$ for every $\alpha,\beta>1$, every $m$, and every $k$.

**Answer 3.3:** In Model A, the evidence $P(S)$ depends on $S$ only through $k$.

**Answer 3.4:** In Model B, the naive Bayes assumption implies $P(x_j,x_{j'})=P(x_j)P(x_{j'})$ for all $j\ne j'$.

**Answer 3.5:** In Model B, the set of class-conditional distributions on $\{0,1\}^d$ representable under the naive Bayes assumption is the set of **all** distributions on $\{0,1\}^d$, for every $d\ge1$.

---

## Question 4

Let $Z\in\mathbb{R}^{m\times(d+1)}$ hold the augmented inputs in its rows and let $y\in\mathbb{R}^m$. Define

$$w_{LS}\in\arg\min_{w}\|Zw-y\|_2^2, \qquad w_\lambda := \arg\min_{w}\ \underbrace{\|Zw-y\|_2^2}_{=:\,m\,\widehat R_S(w)}+\lambda\|w\|_2^2, \qquad v_\lambda\in\arg\min_{w}\ \|Zw-y\|_2^2+\lambda\|w\|_1,$$

for $\lambda>0$. It is given that $w_\lambda=(Z^\top Z+\lambda I)^{-1}Z^\top y$. Evaluate the following statements.

**Answer 4.1:** $w_\lambda$ is uniquely defined for every $\lambda>0$, including when $\mathrm{rank}(Z)<d+1$, whereas in that case $\arg\min_w\|Zw-y\|_2^2$ is not a singleton.

**Answer 4.2:** Both $\|w_\lambda\|_2$ and $\widehat R_S(w_\lambda)$ are non-increasing functions of $\lambda$ on $(0,\infty)$.

**Answer 4.3:** For every $(Z,y)$ and every $\lambda>0$, at least one coordinate of $w_\lambda$ equals $0$ exactly.

**Answer 4.4:** Comparing the objective at $w_\lambda$ against the candidate $w=0$ yields $\|w_\lambda\|_2\le\|y\|_2/\sqrt\lambda$; combined with Theorem 5a.4, the resulting bound on the Rademacher complexity of $\{x\mapsto w^\top x : \|w\|_2\le\|y\|_2/\sqrt{\lambda}\}$ is a decreasing function of $\lambda$.

**Answer 4.5:** There exist $(Z,y)$ and $\lambda>0$ for which some coordinate $j$ satisfies $(v_\lambda)_j=0$ and $(w_\lambda)_j\ne0$.

---

## Question 5

Let $\ell$ be the squared loss, $f^*(x):=\mathbb{E}[y\mid x]$, $\|g\|^2:=\mathbb{E}_{x}[g(x)^2]$, $h_S\gets A(S)$, and $\bar h(x):=\mathbb{E}_S[h_S(x)]$. It is given that

$$R(h)-R^*_{\mathcal{D}}=\|h-f^*\|^2, \qquad \mathbb{E}_S[R(h_S)]-R^*_{\mathcal{D}} \;=\; \underbrace{\|\bar h-f^*\|^2}_{\mathrm{Bias}^2}+\underbrace{\mathbb{E}_x\big[\mathrm{Var}_S[h_S(x)]\big]}_{\mathrm{Variance}} \;=\; \epsilon_{app}+\mathbb{E}_S[\epsilon_{est}].$$

Evaluate the following statements.

**Answer 5.1:** The identity $\mathrm{Bias}^2+\mathrm{Variance}=\epsilon_{app}+\mathbb{E}_S[\epsilon_{est}]$ holds for every $\mathcal{H}$ and every $A$, convex or not.

**Answer 5.2:** If $\mathcal{H}$ is convex and $h_S\in\mathcal{H}$ for every $S$, then $\mathrm{Bias}^2\ge\epsilon_{app}$ and $\mathrm{Variance}\le\mathbb{E}_S[\epsilon_{est}]$.

**Answer 5.3:** $R(\bar h)\le\min_{h\in\mathcal{H}}R(h)$ for every $\mathcal{H}$.

**Answer 5.4:** The same additive identity holds for the zero-one loss with the analogous definitions.

**Answer 5.5:** If $\mathcal{H}\subseteq\mathcal{H}'$, then $\epsilon_{app}(\mathcal{H}')\le\epsilon_{app}(\mathcal{H})$ and $\min_{h\in\mathcal{H}'}\widehat R_S(h)\le\min_{h\in\mathcal{H}}\widehat R_S(h)$, for every $\mathcal{D}$ and every $S$.

---

## Question 6

Let $|\mathcal{H}|=N<\infty$ and $\ell$ take values in $[0,1]$. It is given that $m^{\mathrm{UC}}_{\mathcal{H}}(\epsilon,\delta)\le\frac{\log(2N/\delta)}{2\epsilon^2}$ and that with probability at least $1-\delta$, simultaneously for all $h\in\mathcal{H}$,

$$R(h)\le\widehat R_S(h)+\sqrt{\frac{\log N+\log(1/\delta)}{2m}}.$$

Evaluate the following statements.

**Answer 6.1:** Replacing $N$ by $2N$ at fixed $(\epsilon,\delta)$ increases the sufficient sample size given above by the **additive** amount $\frac{\log 2}{2\epsilon^2}$.

**Answer 6.2:** Replacing $\epsilon$ by $\epsilon/2$ at fixed $(N,\delta)$ multiplies the sufficient sample size given above by $4$.

**Answer 6.3:** The displayed bound remains valid if $\mathcal{H}$ is selected after observing $S$, provided $|\mathcal{H}|=N$ still holds.

**Answer 6.4:** Agnostic PAC learnability of $\mathcal{H}$ implies PAC learnability of $\mathcal{H}$, and for binary classification under the zero-one loss the converse implication holds as well.

**Answer 6.5:** If $\mathcal{H}$ is realizable with respect to $\mathcal{D}$, then every $h_S\in\arg\min_{h\in\mathcal{H}}\widehat R_S(h)$ satisfies $\widehat R_S(h_S)=0$ with probability $1$.

---

## Question 7

Let $\mathcal{X}=\mathbb{R}$ and, for $k\ge1$, let

$$\mathcal{H}_k := \Big\{ x\mapsto \mathbb{1}\big(x\in \textstyle\bigcup_{r=1}^{k}[a_r,b_r]\big) \;:\; a_1\le b_1\le a_2\le\cdots\le b_k \Big\}, \qquad \mathcal{H}_\infty := \bigcup_{k\ge1}\mathcal{H}_k .$$

It is given that $d_{VC}(\mathcal{H}_k)=2k$. Evaluate the following statements.

**Answer 7.1:** $\mathcal{H}_1$ shatters some set of $2$ points in $\mathbb{R}$ and shatters no set of $3$ points in $\mathbb{R}$.

**Answer 7.2:** In general, $d_{VC}(\mathcal{H})=d$ means that **every** set of $d$ points in $\mathcal{X}$ is shattered by $\mathcal{H}$.

**Answer 7.3:** $\tau_{\mathcal{H}_1}(m)\le\binom{m}{0}+\binom{m}{1}+\binom{m}{2}$ for every $m\ge2$.

**Answer 7.4:** Every ERM rule over $\mathcal{H}_k$ is an agnostic PAC learner for $\mathcal{H}_k$, for every fixed $k$.

**Answer 7.5:** $d_{VC}(\mathcal{H}_\infty)=\infty$, and consequently $\mathcal{H}_\infty$ is neither PAC learnable nor nonuniformly learnable.

---

## Question 8

Let $\mathcal{H}=\bigcup_{i\in\mathbb{N}}\mathcal{H}_i$ where each $\mathcal{H}_i$ has the uniform convergence property with sample complexity $m^{\mathrm{UC}}_{\mathcal{H}_i}$. Let $w:\mathbb{N}\to[0,1]$, let $\epsilon_i(m,\delta):=\min\{\epsilon\in(0,1): m^{\mathrm{UC}}_{\mathcal{H}_i}(\epsilon,\delta)\le m\}$, let $i(h):=\min\{i : h\in\mathcal{H}_i\}$, and let

$$h_{SRM}\in\arg\min_{h\in\mathcal{H}}\Big\{\widehat R_S(h)+\epsilon_{i(h)}\big(m,\ w(i(h))\,\delta\big)\Big\}.$$

Evaluate the following statements.

**Answer 8.1:** Theorem 5.7 remains valid under the weaker hypothesis $\lim_{i\to\infty}w(i)=0$ in place of $\sum_{i}w(i)<1$.

**Answer 8.2:** If $w$ is permitted to depend on $S$, the union-bound step in the proof of Theorem 5.7 is no longer valid.

**Answer 8.3:** For a prefix-free description language $s(\cdot)$ over $\{0,1\}$, the choice $w'(h):=2^{-|s(h)|}$ satisfies the summability requirement, and this is exactly what Kraft's inequality certifies.

**Answer 8.4:** Theorem 5.4 determines the constants $C_1$ and $C_2$, so any two classes with the same finite VC dimension have the same sample complexity at the same $(\epsilon,\delta)$.

**Answer 8.5:** Every nonuniformly learnable class is agnostic PAC learnable.

---

## Question 9

For $A\subset\mathbb{R}^m$ and i.i.d. Rademacher signs $\sigma_1,\ldots,\sigma_m$, let $\widehat{\mathfrak{R}}(A):=\mathbb{E}_\sigma\big[\sup_{a\in A}\frac1m\sum_{i=1}^m\sigma_i a_i\big]$; let $\mathcal{H}_S:=\{(h(x_1),\ldots,h(x_m)):h\in\mathcal{H}\}$ and $\widehat{\mathfrak{R}}_S(\mathcal{H}):=\widehat{\mathfrak{R}}(\mathcal{H}_S)$. Evaluate the following statements.

**Answer 9.1:** If $A=\{a\}$ is a singleton then $\widehat{\mathfrak{R}}(A)=0$, and if $A\subseteq A'$ then $\widehat{\mathfrak{R}}(A)\le\widehat{\mathfrak{R}}(A')$.

**Answer 9.2:** $\widehat{\mathfrak{R}}_S(\mathcal{H})$ depends on the labels $y_1,\ldots,y_m$ of $S$.

**Answer 9.3:** $\widehat{\mathfrak{R}}\big(\mathrm{conv}(A)\big)=\widehat{\mathfrak{R}}(A)$ for every $A\subset\mathbb{R}^m$, where $\mathrm{conv}(A)$ is the set of finite convex combinations of elements of $A$.

**Answer 9.4:** The contraction lemma inequality $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})\le\rho\,\widehat{\mathfrak{R}}_S(\mathcal{H})$, for a $\rho$-Lipschitz $\ell(y,\cdot)$, requires $\mathcal{H}$ to be convex.

**Answer 9.5:** Theorem 5a.3 bounds $\widehat{\mathfrak{R}}_S(\mathcal{H})$ in terms of $d_{VC}(\mathcal{H})$; conversely, if $\widehat{\mathfrak{R}}_S(\mathcal{H})\le c/\sqrt m$ holds for a constant $c$ and every sample $S$ of every size $m$, then $d_{VC}(\mathcal{H})<\infty$.

---

## Question 10

Let $\mathrm{dist}(h,h'):=\mathbb{E}_{x\sim\mathcal{D}_{\mathcal{X}}}|h(x)-h'(x)|$, let $\ell(y,\cdot)$ be $G$-Lipschitz with values in $[0,1]$, and let $N(\epsilon):=N(\epsilon,\mathcal{H},\mathrm{dist})$. It is given (Theorem 5a.5) that for a **fixed** $\epsilon>0$, with probability at least $1-\delta$, simultaneously for all $h\in\mathcal{H}$,

$$R(h)\le\widehat R_S(h)+2G\epsilon+\sqrt{\frac{\log\big(2N(\epsilon)/\delta\big)}{2m}},$$

and (Lemma 5a.3) that $N(\epsilon,\mathcal{B}_B,\|\cdot\|_2)\le(1+2B/\epsilon)^d$ for the Euclidean ball $\mathcal{B}_B\subset\mathbb{R}^d$ of radius $B$. It is further given (Theorem 5a.6) that for a prior $q$ fixed before $S$ and a **fixed** $\beta>0$, with probability at least $1-\delta$, simultaneously for every posterior $\rho$,

$$R(\rho)\le\widehat R_S(\rho)+\tfrac1\beta \mathrm{KL}(\rho\|q)+\tfrac1\beta\log\tfrac1\delta+\tfrac{\beta}{8m}.$$

Evaluate the following statements.

**Answer 10.1:** As stated, the covering bound holds simultaneously for all $\epsilon>0$ on a single event of probability at least $1-\delta$.

**Answer 10.2:** For an infinite $\mathcal{H}$, letting $\epsilon\to0$ sends $2G\epsilon\to0$ but $\log N(\epsilon)\to\infty$, so the right-hand side is not minimized at $\epsilon=0$.

**Answer 10.3:** By Lemma 5a.3, $\log N(\epsilon,\mathcal{B}_B,\|\cdot\|_2)$ grows linearly in $d$, even though $N(\epsilon,\mathcal{B}_B,\|\cdot\|_2)$ itself grows exponentially in $d$.

**Answer 10.4:** The PAC-Bayes bound above remains valid when $\beta$ is chosen after observing $S$, for instance by minimizing the right-hand side over $\beta$.

**Answer 10.5:** Recovering Theorem 5.1 from Theorem 5a.6, via a uniform prior on a finite $\mathcal{H}$ and Dirac posteriors, requires a value of $\beta$ that depends on $S$.

---

## Question 11

Consider the following recursive procedure, with hyperparameters $D_{\max}\in\mathbb{N}\cup\{\infty\}$ and $n_{\min}\ge1$, an impurity measure $\mathrm{Imp}\in\{H,G\}$, and $\mathrm{Gain}(S_v,\phi):=\mathrm{Imp}(S_v)-\frac{|S_L|}{|S_v|}\mathrm{Imp}(S_L)-\frac{|S_R|}{|S_v|}\mathrm{Imp}(S_R)$.

```
GrowTree(S_v, depth):
  1. if depth = D_max  or  |S_v| < n_min  or  Imp(S_v) = 0:
         return Leaf( majority(S_v) )
  2. (j*, t*) := argmax over all (j,t) whose split leaves both children non-empty
                 of  Gain(S_v, phi_{j,t}),   where phi_{j,t}(x) = 1(x_j <= t)
  3. if Gain(S_v, phi_{j*,t*}) <= 0:
         return Leaf( majority(S_v) )
  4. S_L := { (x,y) in S_v : x_j* <= t* };   S_R := S_v \ S_L
  5. return Node(j*, t*, GrowTree(S_L, depth+1), GrowTree(S_R, depth+1))
```

Throughout, take $D_{\max}=\infty$, $n_{\min}=1$, and assume that no two training points share a feature vector while carrying different labels. Evaluate the following statements.

**Answer 11.1:** Under these settings the procedure returns a tree $T$ with $\widehat R_S(h_T)=0$.

**Answer 11.2:** If line 3 is deleted, the procedure still terminates and then returns a tree $T$ with $\widehat R_S(h_T)=0$.

**Answer 11.3:** Replacing $G$ by $H$ in $\mathrm{Imp}$ can change the pair $(j^*,t^*)$ selected at line 2, whereas replacing the base of the logarithm inside $H$ cannot.

**Answer 11.4:** With a finite $D_{\max}$, the number of leaves of the returned tree is at most $2^{D_{\max}}$, so the output always lies in a subclass $\mathcal{H}_{\mathrm{tree},L}$ of finite VC dimension.

**Answer 11.5:** Because line 2 maximizes $\mathrm{Gain}$ at each node, the returned tree minimizes $\widehat R_S$ over all trees with the same number of leaves.

---

## Question 12

**Setup A (AdaBoost).** $\mathcal{Y}=\{-1,+1\}$, base class $\mathcal{H}_0$, weak learner $\mathrm{WL}$.

```
1. D^(1) := (1/m, ..., 1/m)
2. for t = 1..T:
     2.1  h_t := WL(S, D^(t))
     2.2  eps_t := sum_i D_i^(t) * 1( h_t(x_i) != y_i )
     2.3  w_t := (1/2) log( (1 - eps_t) / eps_t )
     2.4  D_i^(t+1) := D_i^(t) exp( -w_t y_i h_t(x_i) ) / Z,   Z the normalizer
3. output h_S := sign(f_T),  f_T := sum_t w_t h_t
```

It is given that $\widehat R_S(h_S)\le \prod_{t=1}^T 2\sqrt{\epsilon_t(1-\epsilon_t)}$, and that $2\sqrt{u(1-u)}\le1$ for every $u\in[0,1]$.

**Setup B (bagging).** $h_{S_1},\ldots,h_{S_T}$ with $S_1,\ldots,S_T\overset{\text{i.i.d.}}{\sim}\mathcal{D}^m$ independent, versus $\tilde h_{S_1},\ldots,\tilde h_{S_T}$ trained on $T$ bootstrap resamples of one fixed $S$.

Evaluate the following statements.

**Answer 12.1:** If $\epsilon_t=1/2$ at some round $t$, then $w_t=0$ and $D^{(t+1)}=D^{(t)}$.

**Answer 12.2:** If $0<\epsilon_t<1/2$, then the weighted error of the **same** $h_t$ measured under $D^{(t+1)}$ equals exactly $1/2$.

**Answer 12.3:** If $\epsilon_t\le\frac12-\gamma$ holds at $T/2$ of the $T$ rounds and only $\epsilon_t\le\frac12$ is known at the rest, then $\widehat R_S(h_S)\le e^{-\gamma^2 T}$.

**Answer 12.4:** Since $\prod_{t\le T}2\sqrt{\epsilon_t(1-\epsilon_t)}$ is non-increasing in $T$, the quantity $R(h_S)$ is non-increasing in $T$ as well.

**Answer 12.5:** In Setup B, $\mathrm{Var}\big[\frac1T\sum_t h_{S_t}(x)\big]=\frac1T\mathrm{Var}_S[h_S(x)]$ for every fixed $x$, whereas the corresponding equality can fail for $\frac1T\sum_t \tilde h_{S_t}(x)$.

---

## Question 13

**Setup A.** $h_{W,w}(x)=w^\top g(Wx)$ with $W\in\mathbb{R}^{k\times d}$, $w\in\mathbb{R}^k$, $g$ applied elementwise, $1$-Lipschitz and satisfying $g(0)=0$; every row of $W$ has $\|w^{(1)}_j\|_2\le B_1$; $\|w\|_2\le B_2$; $\|x\|_2\le R$ almost surely. Write $\mathcal{H}_{k,B_1,B_2}$ for this class. Theorem 8.1 gives $\widehat{\mathfrak{R}}_S(\mathcal{H}_{k,B_1,B_2})\le \sqrt k\,B_1B_2R/\sqrt m$.

**Setup B.** $\mathcal{H}_k$ is the RKHS of a kernel $k$, $\Psi:\mathbb{R}^m\to\mathbb{R}$ is arbitrary, $\mathrm{pen}:[0,\infty)\to\mathbb{R}$, and

$$f^*\in\arg\min_{f\in\mathcal{H}_k}\ \Psi\big(f(x_1),\ldots,f(x_m)\big)+\mathrm{pen}\big(\|f\|_{\mathcal{H}_k}\big).$$

Evaluate the following statements.

**Answer 13.1:** The bound of Theorem 8.1 depends on neither the input dimension $d$ nor the width $k$.

**Answer 13.2:** Theorem 8.1 as stated applies to $g=\tanh$ but not, without modification, to $g(z)=1/(1+e^{-z})$.

**Answer 13.3:** Theorem 8.1 upper bounds $R(h)-\widehat R_S(h)$ for every $h\in\mathcal{H}_{k,B_1,B_2}$, but it does not assert that gradient descent returns an $h$ with small $\|w\|_2$ and $\|w^{(1)}_j\|_2$.

**Answer 13.4:** If $\mathrm{pen}$ is strictly increasing, then **every** minimizer $f^*$ lies in a subspace of $\mathcal{H}_k$ of dimension at most $m$, whatever the dimension of $\mathcal{H}_k$.

**Answer 13.5:** The conclusion of 13.4 continues to hold when $\mathrm{pen}$ is only assumed non-decreasing.

---

# Answer Key

*(To be removed from the version handed to students.)*

| | .1 | .2 | .3 | .4 | .5 |
|---|---|---|---|---|---|
| **Q1** | True | True | True | True | False |
| **Q2** | True | False | True | False | True |
| **Q3** | True | False | True | False | False |
| **Q4** | True | False | False | True | True |
| **Q5** | True | True | False | False | True |
| **Q6** | True | True | False | True | True |
| **Q7** | True | False | True | True | False |
| **Q8** | False | True | True | False | False |
| **Q9** | True | False | True | False | False |
| **Q10** | False | True | True | False | False |
| **Q11** | False | True | True | True | False |
| **Q12** | True | True | True | False | True |
| **Q13** | False | True | True | True | False |

## Proofs and counterexamples

**Q1.** *1.1* Each $x_i$ satisfies $h_S(x_i)=y_i$ by construction. *1.2* $\{x_1,\ldots,x_m\}$ is finite, hence $\mathcal{D}_{\mathcal{X}}$-null, so $h_S(x)=0$ almost surely and $R(h_S)=P(f(x)\ne0)=q$. *1.3* $\widehat R_S(h_S)=0$ is minimal; any $h$ agreeing with $f$ on $\{x_i\}$ also attains $0$, and there are infinitely many such $h$ since $\mathcal{X}$ is infinite. *1.4* Take $h(x)=f(x)$ on $\{x_i\}$ and $h(x)=1-f(x)$ elsewhere: $\widehat R_S(h)=0$ and $R(h)=1$. *1.5* **False.** Lemma 5.1 is conditional on $S$ being $\epsilon/2$-representative; statement 1.4 shows that for $\mathcal{H}_{\mathrm{all}}$ this hypothesis fails maximally, for every $m$. This is the concrete face of Corollary 5.2.

**Q2.** *2.1* Chebyshev with $\mathrm{Var}[\widehat\mu_m]=\sigma^2/m$: $P(|\widehat\mu_m-\mu|\ge\epsilon)\le\sigma^2/(m\epsilon^2)$; setting this to $\delta$ gives $\epsilon=B_{\mathrm{Cheb}}$. *2.2* **False.** Hoeffding's inequality uses boundedness only; $\sigma^2$ never appears. *2.3* $B_{\mathrm{Cheb}}/B_{\mathrm{Hoef}}=\sigma\sqrt{2/(\delta\log(1/\delta))}$, free of $m$ — both bounds shrink at rate $m^{-1/2}$, and Hoeffding's advantage lies entirely in its dependence on $\delta$. *2.4* **False.** The proof of Theorem 5.1 uses the union bound over $h$, which requires no relationship whatsoever between different hypotheses; independence is needed only across $i$ within a fixed $h$. *2.5* Replacing one example changes $\widehat R_S(h)$ by at most $1/m$ for every $h$ at once, hence changes the supremum by at most $1/m$; the argument does not count hypotheses.

**Q3.** *3.1* $\theta_{MAP}=\frac{1+k-1}{1+1+m-2}=k/m=\theta_{MLE}$. *3.2* **False.** The predictive is the posterior *mean* $\frac{\alpha+k}{\alpha+\beta+m}$, the MAP its *mode* $\frac{\alpha+k-1}{\alpha+\beta+m-2}$; these differ unless the posterior is symmetric. *3.3* $P(S\mid\theta)=\theta^k(1-\theta)^{m-k}$ depends on $S$ only through $k$, and $P(S)=\int_0^1 P(S\mid\theta)p(\theta)\,d\theta$ inherits this. *3.4* **False.** The assumption is *conditional* independence given $y$; conditional independence does not imply marginal independence — marginalizing over $y$ generally couples the features. *3.5* **False.** For $d\ge2$ the naive family is a proper subset: it cannot represent any class-conditional in which two coordinates are dependent.

**Q4.** *4.1* $Z^\top Z+\lambda I\succ0$ for $\lambda>0$, so the ridge minimizer is unique; if $\mathrm{rank}(Z)<d+1$ the unregularized objective is constant along $\ker(Z)$. *4.2* **False.** $\|w_\lambda\|_2$ is non-increasing, but $\widehat R_S(w_\lambda)$ is non-*decreasing*: comparing the two optimality conditions for $\lambda_1<\lambda_2$ gives $\|w_{\lambda_1}\|_2\ge\|w_{\lambda_2}\|_2$, and substituting that back yields $\|Zw_{\lambda_1}-y\|^2\le\|Zw_{\lambda_2}-y\|^2$. *4.3* **False.** Ridge shrinks without sparsifying; a generic $(Z,y)$ gives all coordinates non-zero. *4.4* $\lambda\|w_\lambda\|^2\le\|Zw_\lambda-y\|^2+\lambda\|w_\lambda\|^2\le\|y\|^2$; Theorem 5a.4 then gives $BR/\sqrt m$ with $B=\|y\|_2/\sqrt\lambda$, decreasing in $\lambda$. *4.5* This is the defining difference between the two penalties, illustrated by the two weight bar plots of Chapter 2.

**Q5.** *5.1* Both sides equal $\mathbb{E}_S[R(h_S)]-R^*_{\mathcal{D}}$ by the given identities; convexity is nowhere used. *5.2* Convexity puts $\bar h\in\mathcal{H}$, so $\|\bar h-f^*\|^2\ge\min_{h\in\mathcal{H}}\|h-f^*\|^2=\epsilon_{app}$; the second claim follows by subtracting from 5.1. *5.3* **False.** For convex $\mathcal{H}$, $\bar h\in\mathcal{H}$ forces $R(\bar h)\ge\min_{h\in\mathcal{H}}R(h)$. (The reverse inequality *can* hold for non-convex $\mathcal{H}$ — that is the random forest mechanism — but the claim asserts it for every $\mathcal{H}$.) *5.4* **False.** Step 1 of the bridge argument turns excess risk into a squared distance using the squared loss specifically; no additive analogue exists for the zero-one loss. *5.5* Both are minima taken over a larger set.

**Q6.** *6.1* $\frac{\log(2\cdot 2N/\delta)}{2\epsilon^2}-\frac{\log(2N/\delta)}{2\epsilon^2}=\frac{\log2}{2\epsilon^2}$. *6.2* The expression carries $\epsilon^{-2}$, so the factor is $4$. *6.3* **False.** The union bound is taken over a class fixed before $S$; a data-selected class of the same cardinality is not covered — this is the same failure as choosing $h$ after $S$ in 2.4. *6.4* The forward implication is immediate (a realizable $\mathcal{D}$ has $\min_h R(h)=0$); the converse is part of the equivalence in Theorem 5.4. *6.5* Realizability gives some $h^*\in\mathcal{H}$ with $R(h^*)=0$, hence $\widehat R_S(h^*)=0$ almost surely, so the minimum over $\mathcal{H}$ is $0$.

**Q7.** *7.1* $d_{VC}(\mathcal{H}_1)=2$ is exactly this statement. *7.2* **False.** Finite VC dimension $d$ asserts that *some* set of $d$ points is shattered; e.g. for $\mathcal{H}_1$ no set of $3$ collinear points is shattered even though $d_{VC}=2$. *7.3* Lemma 5.2 with $d=2$. *7.4* $d_{VC}(\mathcal{H}_k)=2k<\infty$, so Theorem 5.4 applies. *7.5* **False** in its second half. $d_{VC}(\mathcal{H}_\infty)=\infty$ does rule out PAC learnability (Theorem 5.3), but $\mathcal{H}_\infty$ is a countable union of classes each having uniform convergence, so Theorem 5.6 makes it nonuniformly learnable. This is precisely the gap that nonuniform learnability was introduced to occupy.

**Q8.** *8.1* **False.** The proof bounds $\sum_i P(B_i)\le\delta\sum_i w(i)$, which needs summability; $w(i)=1/i$ tends to $0$ yet diverges. *8.2* A union bound requires the failure budget to be allocated before the event is observed; a data-dependent allocation invalidates it — the same discipline the PAC-Bayes prior obeys in 10.4. *8.3* Kraft's inequality states exactly $\sum_{s}2^{-|s|}\le1$ for a prefix-free set of strings. *8.4* **False.** The theorem brackets the sample complexity between $C_1$ and $C_2$ multiples of one expression and leaves both constants unspecified. *8.5* **False.** The inclusion runs the other way: agnostic PAC learnability implies nonuniform learnability, and $\mathcal{H}_\infty$ of Question 7 is a counterexample to the converse.

**Q9.** *9.1* For a singleton the supremum is inert and $\mathbb{E}[\sigma_i]=0$; monotonicity is monotonicity of a supremum. *9.2* **False.** $\widehat{\mathfrak{R}}_S(\mathcal{H})$ is defined on the restriction $\mathcal{H}_S$, which involves the inputs only. (It is $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$ that sees the labels.) *9.3* Theorem 10.2: linearity of the inner product bounds a convex combination by the best of its constituents. *9.4* **False.** The contraction lemma's proof uses only the Lipschitz property coordinate by coordinate; $\mathcal{H}$ is arbitrary. *9.5* **False.** Theorem 5a.4 exhibits $\{x\mapsto w^\top x:\|w\|_2\le B\}$ on a bounded domain, whose Rademacher complexity is $\le BR/\sqrt m$ for every $S$ while its VC dimension is infinite in an infinite-dimensional feature space. Rademacher complexity is strictly the finer measure.

**Q10.** *10.1* **False.** The bound is stated for a fixed $\epsilon$; simultaneity over $\epsilon$ requires a further union bound over a grid of resolutions, which is exactly the step chaining performs systematically. *10.2* The two terms move in opposite directions, which is why $\epsilon$ is a parameter to be balanced. *10.3* $\log(1+2B/\epsilon)^d=d\log(1+2B/\epsilon)$; it is $\log N$, not $N$, that enters the bound. *10.4* **False.** Fixing $\beta$ before $S$ is what makes the event $(\ast)$ of the proof independent of the later choice of $\rho$; optimizing $\beta$ on the data needs a union bound over $\beta$. *10.5* **False.** The optimizing value $\beta^*=\sqrt{8m(\log N+\log(1/\delta))}$ depends only on $N$, $m$ and $\delta$, so it is a legitimate a-priori choice — which is why the recovery of Theorem 5.1 is valid.

**Q11.** *11.1* **False.** Counterexample (XOR): $S_v=\{((0,0),0),((0,1),1),((1,0),1),((1,1),0)\}$ has $G(S_v)=1/2$, and every admissible split on either feature sends one point of each class to each child, giving weighted impurity $\tfrac12\cdot\tfrac12+\tfrac12\cdot\tfrac12=1/2$ and $\mathrm{Gain}=0$. Line 3 therefore returns a single leaf with $\widehat R_S=1/2>0$. Greedy growth is a surrogate for ERM, not ERM. *11.2* Without line 3 the procedure splits anyway; each admissible split leaves both children non-empty, so node sizes strictly decrease and the recursion terminates, ending at leaves that are pure by line 1. On the XOR node, splitting on either feature yields children that are separated by the other feature. *11.3* $G$ and $H$ are different functions and can rank splits differently; changing the logarithm's base multiplies $H$ by a positive constant, which rescales every $\mathrm{Gain}$ by the same constant and cannot change an $\arg\max$. *11.4* A binary tree of depth at most $D_{\max}$ has at most $2^{D_{\max}}$ leaves, and Proposition 7.1 gives $d_{VC}(\mathcal{H}_{\mathrm{tree},L})=O(L\log(Ld))<\infty$. *11.5* **False.** The counterexample of 11.1 already gives a $1$-leaf output where a $4$-leaf tree, and indeed some larger-gain non-greedy split ordering, does better; exact ERM over trees is NP-hard.

**Q12.** *12.1* $w_t=\tfrac12\log1=0$, so the update multiplies every weight by $e^0=1$ and the normalizer is $1$. *12.2* With $e^{w_t}=\sqrt{(1-\epsilon_t)/\epsilon_t}$, the unnormalized misclassified mass is $\epsilon_te^{w_t}=\sqrt{\epsilon_t(1-\epsilon_t)}$ and the correctly classified mass is $(1-\epsilon_t)e^{-w_t}=\sqrt{\epsilon_t(1-\epsilon_t)}$; they are equal, so after normalization the error of $h_t$ under $D^{(t+1)}$ is $1/2$. Each round makes its own hypothesis useless for the next. *12.3* Every factor satisfies $2\sqrt{\epsilon_t(1-\epsilon_t)}\le1$, and the $T/2$ good rounds contribute $\sqrt{1-4\gamma^2}\le e^{-2\gamma^2}$ each, so the product is at most $e^{-2\gamma^2\cdot T/2}=e^{-\gamma^2T}$. *12.4* **False.** The product bounds the *empirical* risk; nothing in it constrains $R(h_S)$, whose behaviour in $T$ is governed by the margin bound and can be non-monotone. *12.5* Independence makes all cross-covariances vanish (Proposition 10.1); bootstrap resamples share the underlying $S$, so the members are correlated and the variance reduction is generally weaker than $1/T$.

**Q13.** *13.1* **False.** The bound is free of $d$ but carries $\sqrt k$. *13.2* $\tanh$ is $1$-Lipschitz with $\tanh(0)=0$; the sigmoid has $g(0)=1/2\ne0$, so the offset must be carried through the contraction step separately. *13.3* The bound is uniform over the class and says nothing about which member an optimizer reaches — this is exactly the gap that the implicit-bias literature addresses. *13.4* Theorem 9.3: the orthogonal complement of $\mathrm{span}\{k(x_i,\cdot)\}_{i=1}^m$ is invisible to $\Psi$ by the reproducing property and strictly penalized by $\mathrm{pen}$, so it must vanish at any minimizer; that span has dimension at most $m$. *13.5* **False.** With $\mathrm{pen}$ constant, any $f=f_\parallel+f_\perp$ with $f_\perp\ne0$ attains the same objective value as $f_\parallel$ and is therefore also a minimizer. Strict monotonicity is what upgrades "some minimizer has the representer form" to "every minimizer does".
