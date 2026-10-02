Chapter 5 built the entire theory of generalization on top of two concrete complexity measures: the raw count $|\mathcal{H}|$ for finite classes, and the VC dimension $d_{VC}(\mathcal{H})$ for infinite ones. Both did the same job: they compressed a hypothesis class down to a single number that a Hoeffding-type concentration inequality, combined with the union bound, could feed on. This chapter makes that role explicit and abstract, then instantiates it three more times.

**The abstract recipe.** Every bound in this course, past and future, has the shape

$$\begin{gathered}
R(h) \leq \widehat{R}_S(h) + \underbrace{\mathrm{Complexity}(\mathcal{H}, m)}_{\text{shrinks as } m \to \infty} + \underbrace{\mathrm{Confidence}(\delta, m)}_{\text{shrinks as } m \to \infty}, \\
\forall h \in \mathcal{H}, \text{ w.p.} \geq 1-\delta,
\end{gathered}$$

i.e. a bound on the **representativeness** gap $\sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ from Definition 5.4. What is negotiable is only how $\mathrm{Complexity}(\mathcal{H},m)$ is defined. We already have two instances: $\mathrm{Complexity} = \sqrt{\log|\mathcal{H}|/(2m)}$ (Theorem 5.1) and $\mathrm{Complexity} = \sqrt{d_{VC}(\mathcal{H})/m}$ up to logarithmic factors (Theorem 5.4). We now add three more: **Rademacher complexity**, **covering numbers**, and **PAC-Bayes bounds**. All five are, in a precise sense we make explicit below, the same quantity viewed through different lenses — and by the end of this chapter, we will have shown algebraically that the very first bound of Chapter 5 (Theorem 5.1) is a special case of the PAC-Bayes bound derived last.

# A Concentration Tool for Suprema: McDiarmid's Inequality

Hoeffding's inequality (Chapter 4) concentrates a *single* average $\widehat\mu_m$ around its mean. Bounding $\sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ requires concentrating a *function of the whole sample* — the supremum over (possibly infinitely many) hypotheses — around its mean. The right tool is a generalization, due to [McDiarmid (1989)](https://doi.org/10.1017/CBO9781107359949.008)<!-- cite: mcdiarmid1989method | article | author={McDiarmid, Colin}; title={On the Method of Bounded Differences}; journal={Surveys in Combinatorics}; year={1989}; volume={141}; pages={148--188} -->, of Hoeffding's inequality to arbitrary functions of independent random variables, provided the function does not depend too sensitively on any single one of them.

**Theorem 5a.1 (McDiarmid's bounded differences inequality).** Let $Z_1,\ldots,Z_m$ be independent random variables and let $f: \mathcal{Z}^m \to \mathbb{R}$ satisfy the **bounded differences** property: for every $i \in [m]$ and every $z_1,\ldots,z_m,z_i' \in \mathcal{Z}$,

$$\big| f(z_1,\ldots,z_i,\ldots,z_m) - f(z_1,\ldots,z_i',\ldots,z_m) \big| \leq c_i.$$

Then for any $t > 0$,

$$P\big(|f(Z_1,\ldots,Z_m) - \mathbb{E}[f(Z_1,\ldots,Z_m)]| \geq t\big) \leq 2\exp\Big(\frac{-2t^2}{\sum_{i=1}^m c_i^2}\Big).$$

**Proof.** The argument is the Chernoff route of Theorem 4.4, applied not to the summands themselves — there are none — but to an artificial decomposition of $f$ into $m$ increments, one per input variable. Write $Z_{1:i} := (Z_1,\ldots,Z_i)$, $F := f(Z_1,\ldots,Z_m)$, and define the **increments**

$$\begin{gathered}
V_i := \mathbb{E}\big[F \mid Z_{1:i}\big] - \mathbb{E}\big[F \mid Z_{1:i-1}\big], \\
i = 1,\ldots,m,
\end{gathered}$$

with the convention $\mathbb{E}[F \mid Z_{1:0}] := \mathbb{E}[F]$. The sum telescopes exactly:

$$\sum_{i=1}^m V_i = \mathbb{E}[F\mid Z_{1:m}] - \mathbb{E}[F] = F - \mathbb{E}[F].$$

*Step 1: each increment has conditional mean zero and conditional range at most $c_i$.* The tower property gives $\mathbb{E}[V_i \mid Z_{1:i-1}] = 0$. For the range, define
$$
\begin{aligned}
U_i &:= \sup_{z}\ \mathbb{E}\big[F \mid Z_{1:i-1}, Z_i = z\big] - \mathbb{E}[F\mid Z_{1:i-1}], \\
L_i &:= \inf_{z}\ \mathbb{E}\big[F \mid Z_{1:i-1}, Z_i = z\big] - \mathbb{E}[F\mid Z_{1:i-1}],
\end{aligned}
$$
both functions of $Z_{1:i-1}$ alone, so that $L_i \leq V_i \leq U_i$ by construction. For any two values $z, z'$, independence of the $Z$'s lets us average the bounded-differences hypothesis over the *remaining* variables $Z_{i+1},\ldots,Z_m$:
$$\big|\mathbb{E}[F\mid Z_{1:i-1}, Z_i = z] - \mathbb{E}[F\mid Z_{1:i-1}, Z_i = z']\big| \leq \mathbb{E}\big[\,\big|f(\ldots,z,\ldots) - f(\ldots,z',\ldots)\big|\,\big] \leq c_i,$$
hence $U_i - L_i \leq c_i$: conditionally on the past, $V_i$ is a mean-zero random variable confined to an interval of width at most $c_i$.

*Step 2: conditional Hoeffding's lemma, iterated.* Lemma 4.1 applies verbatim to conditional expectations, so for every $s \in \mathbb{R}$,
$$\mathbb{E}\big[e^{sV_i} \mid Z_{1:i-1}\big] \leq e^{s^2c_i^2/8}.$$
Peeling the increments off one at a time with the tower property,
$$\mathbb{E}\big[e^{s\sum_{i=1}^m V_i}\big] = \mathbb{E}\Big[e^{s\sum_{i<m}V_i}\ \underbrace{\mathbb{E}\big[e^{sV_m}\mid Z_{1:m-1}\big]}_{\leq\ e^{s^2c_m^2/8}}\Big] \leq \cdots \leq \exp\Big(\frac{s^2}{8}\sum_{i=1}^m c_i^2\Big).$$

*Step 3: Chernoff.* Exactly as in Theorem 4.4, Markov's inequality on $e^{s(F-\mathbb{E}[F])}$ gives $P(F - \mathbb{E}[F] \geq t) \leq \exp(-st + \tfrac{s^2}{8}\sum_i c_i^2)$ for every $s>0$; minimizing the exponent at $s^* = 4t/\sum_i c_i^2$ yields $\exp\big(-2t^2/\sum_i c_i^2\big)$. Applying the same argument to $-f$ bounds the other tail identically, and adding the two gives the factor $2$ $\square$

Setting $m=1$, or taking $f(z_1,\ldots,z_m) = \frac1m\sum_i z_i$ with $c_i = (b_i-a_i)/m$, recovers Hoeffding's inequality — McDiarmid's inequality is its strict generalization, obtained by replacing "sum of independent variables" with "any function that no single variable can move very far".

For the choice $f(S) := \sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ with a loss bounded in $[0,1]$, changing a single training example changes $\widehat{R}_S(h)$ by at most $1/m$ for every $h$ simultaneously, hence $f$ has bounded differences $c_i = 1/m$, and McDiarmid's inequality applies with $\sum_i c_i^2 = 1/m$. This is exactly the role Hoeffding's inequality played for a single hypothesis in Chapter 4 — McDiarmid is its supremum-over-a-class counterpart, and it is what turns a bound on $\mathbb{E}_S[\sup_h(R(h)-\widehat{R}_S(h))]$ into the high-probability statements below.

# The Symmetrization Technique

The remaining question is how to bound $\mathbb{E}_S\big[\sup_{h\in\mathcal{H}} (R(h)-\widehat{R}_S(h))\big]$ itself. Every complexity measure in this chapter is obtained by the same device: introduce an independent **ghost sample** $S' = \{(x_i',y_i')\}_{i=1}^m$, drawn i.i.d. from $\mathcal{D}$ exactly like $S$, and note that $R(h) = \mathbb{E}_{S'}[\widehat{R}_{S'}(h)]$. Then, since $\sup$ of an expectation is at most the expectation of the $\sup$ (Jensen),

$$
\begin{aligned}
\mathbb{E}_S\Big[\sup_{h\in\mathcal{H}}\big(R(h) - \widehat{R}_S(h)\big)\Big] &= \mathbb{E}_S\Big[\sup_{h\in\mathcal{H}} \mathbb{E}_{S'}\big[\widehat{R}_{S'}(h) - \widehat{R}_S(h)\big]\Big] \\
&\leq \mathbb{E}_{S,S'}\Big[\sup_{h\in\mathcal{H}} \big(\widehat{R}_{S'}(h) - \widehat{R}_S(h)\big)\Big] \\
&= \mathbb{E}_{S,S'}\Big[\sup_{h\in\mathcal{H}} \frac{1}{m}\sum_{i=1}^m \big(\ell(y_i', h(x_i')) - \ell(y_i, h(x_i))\big)\Big].
\end{aligned}
$$

The last expression is symmetric in the pairing $(x_i,y_i) \leftrightarrow (x_i',y_i')$: since $S$ and $S'$ are i.i.d. and independent of each other, swapping the $i$-th pair of $S$ with the $i$-th pair of $S'$ leaves the joint distribution of $(S,S')$ unchanged. Introducing an independent Rademacher variable $\sigma_i \in \{-1,+1\}$ (each sign equally likely) to record whether the $i$-th pair *was* swapped therefore leaves the expectation unchanged for any choice of signs, hence

$$
\begin{aligned}
\mathbb{E}_{S,S'}\Big[\sup_{h}\frac1m\sum_i \big(\ell(y_i', h(x_i'))-\ell(y_i, h(x_i))\big)\Big] &= \mathbb{E}_{S,S',\sigma}\Big[\sup_h \frac1m\sum_i \sigma_i\big(\ell(y_i', h(x_i'))-\ell(y_i, h(x_i))\big)\Big] \\
&\leq 2\, \mathbb{E}_{S,\sigma}\Big[\sup_h \frac1m\sum_i \sigma_i \ell(y_i, h(x_i))\Big],
\end{aligned}
$$

where the last step splits the supremum of a sum into the sum of two suprema (each bounded by the original, by symmetry of $\pm\sigma_i$). This is **symmetrization**, and the quantity on the right is, by definition, twice the (expected) Rademacher complexity of the loss class $\ell \circ \mathcal{H}$.

# Rademacher Complexity

We follow the formulation of [Bartlett & Mendelson (2002)](https://www.jmlr.org/papers/v3/bartlett02a.html)<!-- cite: bartlett2002rademacher | article | author={Bartlett, Peter L. and Mendelson, Shahar}; title={Rademacher and Gaussian Complexities: Risk Bounds and Structural Results}; journal={Journal of Machine Learning Research}; year={2002}; volume={3}; pages={463--482} -->.

**Definition 5a.1 (Rademacher complexity).** For a set $A \subset \mathbb{R}^m$ and i.i.d. Rademacher signs $\sigma_1,\ldots,\sigma_m \in \{-1,+1\}$ ($P(\sigma_i=1)=P(\sigma_i=-1)=\tfrac12$), define

$$\widehat{\mathfrak{R}}(A) := \mathbb{E}_\sigma\Big[\sup_{a \in A} \frac{1}{m}\sum_{i=1}^m \sigma_i a_i\Big].$$

For a hypothesis class $\mathcal{H}$ and a fixed sample $S = \{(x_i,y_i)\}_{i=1}^m$, write $\ell \circ \mathcal{H} \circ S := \big\{(\ell(y_1, h(x_1)),\ldots,\ell(y_m, h(x_m))) : h \in \mathcal{H}\big\} \subset \mathbb{R}^m$ for the loss class evaluated on $S$, and define the **empirical Rademacher complexity** $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) := \widehat{\mathfrak{R}}(\ell\circ\mathcal{H}\circ S)$ — a data-dependent quantity, computable from $S$ alone — and the **(expected) Rademacher complexity** $\mathfrak{R}_m(\ell\circ\mathcal{H}) := \mathbb{E}_S\big[\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})\big]$. When we want the complexity of the hypothesis class itself rather than of the loss class, we write $\widehat{\mathfrak{R}}_S(\mathcal{H}) := \widehat{\mathfrak{R}}(\mathcal{H}_S)$, evaluated on the restriction $\mathcal{H}_S$ of Definition 5.9.

---

Intuitively, $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$ measures how well $\mathcal{H}$ can fit **pure random noise**: $\sigma_i$ is an independent coin flip uncorrelated with everything, so $\sup_h \frac1m\sum_i \sigma_i \ell(y_i, h(x_i))$ asks how large a spurious correlation the richest hypothesis in $\mathcal{H}$ can manufacture with random labels. A class rich enough to realize any loss vector in $\{0,1\}^m$ (as in the memorization hypothesis of Chapter 1) can match every sign, giving $\widehat{\mathfrak{R}}_S \approx \tfrac12$. A class with a single hypothesis has no supremum to exploit, so $\widehat{\mathfrak{R}}_S = \frac{1}{m}\sum_i \mathbb{E}[\sigma_i]\,a_i = 0$ exactly.

**Theorem 5a.2 (Rademacher generalization bound).** Let the loss be bounded in $[0,1]$. For any $\delta \in (0,1)$, with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$, simultaneously for all $h \in \mathcal{H}$:

$$R(h) \leq \widehat{R}_S(h) + 2\,\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + 3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

**Proof.** Write $\Phi(S) := \sup_{h\in\mathcal{H}}\big(R(h)-\widehat{R}_S(h)\big)$, so the claim is a high-probability upper bound on $\Phi(S)$. The proof applies McDiarmid's inequality twice, with symmetrization in between.

*Step 1: concentrate $\Phi$ around its mean.* Replacing one example of $S$ changes $\widehat{R}_S(h)$ by at most $1/m$ for every $h$ at once, hence changes the supremum by at most $1/m$: $\Phi$ has bounded differences $c_i = 1/m$, so $\sum_i c_i^2 = 1/m$. The one-sided form of Theorem 5a.1 at confidence $\delta/2$ gives, with probability at least $1-\delta/2$,
$$\Phi(S) \;\leq\; \mathbb{E}_S[\Phi(S)] + \sqrt{\frac{\log(2/\delta)}{2m}}.$$

*Step 2: bound the mean by the expected Rademacher complexity.* This is the symmetrization argument of the previous section, verbatim: $\mathbb{E}_S[\Phi(S)] \leq 2\,\mathfrak{R}_m(\ell\circ\mathcal{H})$.

*Step 3: replace the expected complexity by its empirical counterpart.* The map $S \mapsto \widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$ also has bounded differences $1/m$: changing one example changes each coordinate of the loss vector by at most $1$, hence changes $\frac1m\sum_i\sigma_i a_i$ by at most $1/m$ uniformly, and a supremum of functions that all move by at most $1/m$ moves by at most $1/m$. Applying Theorem 5a.1 once more at confidence $\delta/2$, with probability at least $1-\delta/2$,
$$\mathfrak{R}_m(\ell\circ\mathcal{H}) = \mathbb{E}_S\big[\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})\big] \;\leq\; \widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + \sqrt{\frac{\log(2/\delta)}{2m}}.$$

*Step 4: union bound.* Both events hold simultaneously with probability at least $1-\delta$, and on their intersection
$$\sup_{h}\big(R(h)-\widehat{R}_S(h)\big) \leq 2\,\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + 2\sqrt{\frac{\log(2/\delta)}{2m}} + \sqrt{\frac{\log(2/\delta)}{2m}},$$
where the factor $2$ from Step 2 multiplies the deviation of Step 3. Rearranging the supremum gives the claim for every $h$ $\square$

This is the Rademacher-complexity instance of the abstract recipe stated at the top of this chapter, with $\mathrm{Complexity}(\mathcal{H},m) = 2\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$.

## Massart's Lemma, and Theorem 5.1 revisited

**Lemma 5a.1 (Massart's lemma).** For a finite set $A = \{a^{(1)},\ldots,a^{(N)}\} \subset \mathbb{R}^m$ with $\|a^{(j)}\|_2 \leq B$ for all $j$,

$$\begin{gathered}
\widehat{\mathfrak{R}}(A) \leq \frac{B\sqrt{2\log N}}{m}, \\
\text{equivalently} \quad \mathbb{E}_\sigma\Big[\max_{j\in[N]}\langle \sigma, a^{(j)}\rangle\Big] \leq B\sqrt{2\log N}.
\end{gathered}$$

**Proof.** Write $X_j := \langle\sigma, a^{(j)}\rangle = \sum_{i=1}^m \sigma_i a_i^{(j)}$ and assume $N \geq 2$ (for $N=1$ the left side is $\mathbb{E}_\sigma[X_1] = 0$ and there is nothing to prove). Fix $s>0$. Because $u \mapsto e^{su}$ is convex and increasing, Jensen's inequality and then the crude bound "a maximum of non-negative terms is at most their sum" give

$$\exp\Big(s\,\mathbb{E}_\sigma\big[\max_j X_j\big]\Big) \;\leq\; \mathbb{E}_\sigma\Big[e^{s \max_j X_j}\Big] = \mathbb{E}_\sigma\Big[\max_j e^{s X_j}\Big] \;\leq\; \sum_{j=1}^N \mathbb{E}_\sigma\big[e^{sX_j}\big].$$

Each $X_j$ is a sum of $m$ independent, mean-zero terms $\sigma_i a_i^{(j)}$, the $i$-th confined to the interval $[-|a_i^{(j)}|, +|a_i^{(j)}|]$ of width $2|a_i^{(j)}|$. Hoeffding's lemma (Lemma 4.1) applied to each term and independence give

$$\mathbb{E}_\sigma\big[e^{sX_j}\big] = \prod_{i=1}^m \mathbb{E}\big[e^{s\sigma_i a_i^{(j)}}\big] \leq \prod_{i=1}^m \exp\Big(\frac{s^2 (2a_i^{(j)})^2}{8}\Big) = \exp\Big(\frac{s^2\|a^{(j)}\|_2^2}{2}\Big) \leq \exp\Big(\frac{s^2B^2}{2}\Big).$$

Substituting and taking logarithms, $s\,\mathbb{E}_\sigma[\max_j X_j] \leq \log N + \tfrac{s^2B^2}{2}$, i.e.

$$\begin{gathered}
\mathbb{E}_\sigma\big[\max_j X_j\big] \leq \frac{\log N}{s} + \frac{sB^2}{2} \\
\text{for every } s>0.
\end{gathered}$$

The right-hand side is minimized at $s^* = \sqrt{2\log N}/B$, where its value is $B\sqrt{2\log N}$. Dividing by $m$ turns the left-hand side into $\widehat{\mathfrak{R}}(A)$ $\square$

Note the shape of the result: the dependence on the number of elements is only $\sqrt{\log N}$, which is what makes the union-bound-style arguments of Chapter 5 as cheap as they are. Note also that the bound is *not* improved by adding a point to $A$ that is far from the others — only the count and the worst-case norm matter.

Apply this to $\ell\circ\mathcal{H}\circ S$ for a **finite** hypothesis class, $N = |\mathcal{H}|$, with loss bounded in $[0,1]$ so $\|a^{(j)}\|_2 \leq \sqrt m$: this gives $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) \leq \sqrt{2\log|\mathcal{H}|/m}$, and plugging into Theorem 5a.2 recovers, up to constants, exactly the finite-hypothesis-class bound of Theorem 5.1. **The very first generalization bound of Chapter 5 was already an instance of a Rademacher-complexity bound** — we simply proved it directly via Hoeffding and the union bound there, without naming the general machinery.

## The Contraction Lemma

**Lemma 5a.2 (Contraction).** If $\phi_i: \mathbb{R} \to \mathbb{R}$ is $\rho$-Lipschitz for each $i$, then $\widehat{\mathfrak{R}}\big(\{(\phi_1(a_1),\ldots,\phi_m(a_m)) : a \in A\}\big) \leq \rho\, \widehat{\mathfrak{R}}(A)$.

**Proof.** The whole content is a **one-coordinate** statement, which we prove first and then apply $m$ times:

> **One-coordinate step.** Let $\psi_1,\ldots,\psi_{m-1}$ be arbitrary functions and $\phi$ be $\rho$-Lipschitz. Then
> $$\mathbb{E}_\sigma\Big[\sup_{a\in A}\Big(\sum_{i<m}\sigma_i\psi_i(a_i) + \sigma_m\phi(a_m)\Big)\Big] \;\leq\; \mathbb{E}_\sigma\Big[\sup_{a\in A}\Big(\sum_{i<m}\sigma_i\psi_i(a_i) + \sigma_m\,\rho\, a_m\Big)\Big].$$

Granting this, apply it to coordinate $m$ (with $\psi_i := \phi_i$), replacing $\phi_m$ by $\rho\cdot\mathrm{id}$; then to coordinate $m-1$ of the resulting set, and so on. After $m$ steps every $\phi_i$ has been replaced by $\rho\cdot\mathrm{id}$, so the left-hand side of the lemma is at most $\widehat{\mathfrak{R}}(\{\rho a : a \in A\}) = \rho\,\widehat{\mathfrak{R}}(A)$, the last equality by homogeneity of the definition. Each step transforms a *different* coordinate, so the factor $\rho$ is collected once in total, not $m$ times.

To prove the one-coordinate step, condition on $\sigma_1,\ldots,\sigma_{m-1}$ and abbreviate $u(a) := \sum_{i<m}\sigma_i\psi_i(a_i)$, which does not involve the $m$-th coordinate's transform. Averaging over the two equally likely values of $\sigma_m$,

$$
\begin{aligned}
\mathbb{E}_{\sigma_m}\Big[\sup_{a \in A}\big(u(a) + \sigma_m \phi(a_m)\big)\Big] &= \tfrac12 \sup_{a\in A}\big(u(a)+\phi(a_m)\big) + \tfrac12\sup_{a'\in A}\big(u(a')-\phi(a_m')\big)\\
&= \tfrac12 \sup_{a,a' \in A}\Big(u(a) + u(a') + \phi(a_m) - \phi(a_m')\Big)\\
&\leq \tfrac12 \sup_{a,a' \in A}\Big(u(a) + u(a') + \rho\,\big|a_m - a_m'\big|\Big)\\
&= \tfrac12 \sup_{a,a' \in A}\Big(u(a) + u(a') + \rho\,\big(a_m - a_m'\big)\Big)\\
&= \mathbb{E}_{\sigma_m}\Big[\sup_{a\in A}\big(u(a) + \sigma_m \rho\, a_m\big)\Big],
\end{aligned}
$$

where the second line merges the two independent suprema into one over the pair $(a,a')$; the third applies the $\rho$-Lipschitz property of $\phi$; the fourth drops the absolute value because the expression is symmetric under swapping $a \leftrightarrow a'$, so the supremum is attained at a pair for which $a_m - a_m' \geq 0$ anyway; and the fifth reverses the first two steps with $\rho\,a_m$ in place of $\phi(a_m)$. Taking the expectation over $\sigma_1,\ldots,\sigma_{m-1}$ gives the one-coordinate step $\square$

Note where the argument would break for the *signed* variant $\sup_a|\frac1m\sum_i\sigma_i a_i|$: the second line merges two suprema into one, which is valid for a supremum of a set but not for a supremum of absolute values — this is the source of the extra factor of $2$ mentioned below.

This lets us pass, in the definition of $\ell \circ \mathcal{H}$, from the Rademacher complexity of the *loss-composed* class directly to the Rademacher complexity of $\mathcal{H}$ itself, whenever the loss $\ell(\cdot,y)$ is $\rho$-Lipschitz in its first argument: $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) \leq \rho\,\widehat{\mathfrak{R}}_S(\mathcal{H})$. A word of caution, since this is a place where different textbooks state slightly different constants: the *clean* version above, with no extra factor, holds for the **unsigned** Rademacher complexity (where the supremum in Definition 5a.1 is *not* wrapped in an absolute value, exactly as we defined it). If instead one works with the signed variant $\sup_a |\frac1m\sum_i\sigma_i a_i|$, the contraction inequality picks up an extra factor of $2$, and the standard proof of even that weaker bound requires $\phi_i(0)=0$ for all $i$. Since our Definition 5a.1 already uses the unsigned form throughout, the clean statement above applies directly to everything in this chapter.

## The Bridge to VC Dimension

**Theorem 5a.3 (VC dimension bounds Rademacher complexity).** For a hypothesis class $\mathcal{H}$ of binary classifiers with $d_{VC}(\mathcal{H}) = d < \infty$, and the zero-one loss,

$$\begin{gathered}
\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq 2\sqrt{\frac{d \log(m+1)}{m}}, \\
\text{for every sample } S \text{ of size } m.
\end{gathered}$$

**Proof sketch.** The restriction $\mathcal{H}_S = \{(h(x_1),\ldots,h(x_m)) : h \in \mathcal{H}\}$ is a *finite* set (Definition 5.9), of size $|\mathcal{H}_S| = \tau_{\mathcal{H}}(m) \leq (m+1)^d$ by the Sauer-Shelah-Perles lemma (Lemma 5.2, using the simplified polynomial form). Apply Massart's lemma (Lemma 5a.1) with $N = (m+1)^d$ and $B=\sqrt m$ (binary values in $\{0,1\}$, so $\|a^{(j)}\|_2 \le \sqrt{m}$):

$$\widehat{\mathfrak{R}}(\mathcal{H}_S) \leq \frac{\sqrt{m}\sqrt{2\log\big((m+1)^d\big)}}{m} = \sqrt{\frac{2 d\log(m+1)}{m}} \leq 2\sqrt{\frac{d\log(m+1)}{m}},$$

the last step simply using $\sqrt 2 \leq 2$, which is where the slack in the stated constant comes from. $\square$

This is the promised unification made concrete: **VC dimension is not a separate theory from Rademacher complexity — it is a specific, combinatorial way of bounding it**, via Sauer-Shelah-Perles turning an infinite hypothesis class into a finite restriction on any given sample, to which Massart's finite-class bound then applies.

## A VC-Dimension-Free Example: Linear Predictors

One reason Rademacher complexity is worth the extra abstraction: it can be **finite and small even when $d_{VC}(\mathcal{H}) = \infty$**, provided we constrain the hypothesis class by a *norm* rather than by dimension.

**Theorem 5a.4.** Let $\mathcal{H} = \{x \mapsto w^\top x : \|w\|_2 \leq B\}$ and suppose $\|x\|_2 \leq R$ almost surely under $\mathcal{D}$. Then

$$\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq \frac{BR}{\sqrt{m}}.$$

**Proof.** By definition and linearity, for a fixed realization of the signs,
$$\sup_{\|w\|_2\leq B} \frac1m \sum_{i=1}^m \sigma_i\, w^\top x_i = \frac1m \sup_{\|w\|_2 \leq B} \Big\langle w, \sum_{i=1}^m \sigma_i x_i\Big\rangle = \frac{B}{m}\Big\|\sum_{i=1}^m \sigma_i x_i\Big\|_2,$$
where the second equality is the equality case of the Cauchy-Schwarz inequality: the linear functional $w \mapsto \langle w, v\rangle$ is maximized over the ball $\|w\|_2\leq B$ at $w = B\,v/\|v\|_2$, with value $B\|v\|_2$. Taking $\mathbb{E}_\sigma$ and applying Jensen's inequality to the concave map $u \mapsto \sqrt u$,
$$\widehat{\mathfrak{R}}_S(\mathcal{H}) = \frac{B}{m}\,\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2\Big] \leq \frac{B}{m}\sqrt{\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2^2\Big]}.$$
Expanding the squared norm and using $\mathbb{E}[\sigma_i\sigma_{i'}] = \mathds{1}(i=i')$ (the signs are independent and mean-zero, and $\sigma_i^2=1$),
$$\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2^2\Big] = \sum_{i,i'} \mathbb{E}[\sigma_i\sigma_{i'}]\, x_i^\top x_{i'} = \sum_{i=1}^m \|x_i\|_2^2 \leq mR^2 .$$
Combining, $\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq \frac{B}{m}\sqrt{mR^2} = \frac{BR}{\sqrt m}$ $\square$

Notice this bound does not depend on the input dimension $d$ at all — only on the norm bound $B$ and the data norm bound $R$. This matters directly for the kernel methods of Chapter 9, where $\phi(x)$ can live in an infinite-dimensional feature space (for which $d_{VC}$ is typically infinite) while $\|w\|_2$ and $\|\phi(x)\|_2$ remain finite: Rademacher complexity, unlike VC dimension, gives a meaningful, finite generalization guarantee in exactly this regime.

# Covering Numbers

Rademacher complexity measures $\mathcal{H}$'s capacity to correlate with noise. **Covering numbers** measure something more literal: how many hypotheses it takes to approximate every hypothesis in $\mathcal{H}$ to a given tolerance.

**Definition 5a.2 ($\epsilon$-cover, covering number).** Let $\mathcal{H}$ be a hypothesis class equipped with a (pseudo-)metric $\mathrm{dist}(\cdot,\cdot)$: a function satisfying axioms M2 and M3 of the metric-space definition of Chapter 2, while M1 is relaxed to the one-directional $\mathrm{dist}(h,h') = 0 \Leftarrow h = h'$, so that distinct hypotheses may sit at distance zero. A finite set $\{h_1,\ldots,h_N\} \subset \mathcal{H}$ is an **$\epsilon$-cover** of $\mathcal{H}$ if for every $h \in \mathcal{H}$ there exists $j \in [N]$ with $\mathrm{dist}(h,h_j) \leq \epsilon$. The **covering number** $N(\epsilon,\mathcal{H},\mathrm{dist})$ is the size of the smallest such cover.

---

In words, an $\epsilon$-cover is a finite "grid" of representative hypotheses such that every hypothesis in $\mathcal{H}$, however large or infinite the class, is within $\epsilon$ of some grid point — turning an infinite class into a finite proxy at the cost of an $\epsilon$-sized approximation error.

The relevant metric for generalization is one under which the risk is Lipschitz: if $\ell(\cdot,y)$ is $G$-Lipschitz in its first argument for every $y$, and $\mathrm{dist}(h,h') := \mathbb{E}_{x\sim\mathcal{D}}|h(x)-h'(x)|$, then $|R(h)-R(h')| \leq G\,\mathrm{dist}(h,h')$ for all $h,h' \in \mathcal{H}$ — replacing $h$ by the nearest cover element changes the risk by only $G\epsilon$. This particular $\mathrm{dist}$ is a genuine pseudo-metric rather than a metric: two distinct hypotheses that differ only on inputs that $\mathcal{D}$ never draws have distance zero, which is exactly why M1 had to be relaxed above — and it is harmless, since such hypotheses also have identical risk.

**Theorem 5a.5 (Covering-number generalization bound).** Let $\{h_1,\ldots,h_{N(\epsilon)}\}$ be an $\epsilon$-cover of $\mathcal{H}$ under $\mathrm{dist}$ as above, with loss bounded in $[0,1]$. Then, for any $h \in \mathcal{H}$, choosing $h_j$ within $\epsilon$ of $h$,

$$
\begin{aligned}
\big|\widehat{R}_S(h) - R(h)\big| &\leq \big|\widehat{R}_S(h) - \widehat{R}_S(h_j)\big| + \big|\widehat{R}_S(h_j)-R(h_j)\big| + \big|R(h_j)-R(h)\big| \\
&\leq 2G\epsilon + \big|\widehat{R}_S(h_j)-R(h_j)\big|,
\end{aligned}
$$

using the Lipschitz bound on both the empirical and true risk. Applying the finite-hypothesis-class bound (Theorem 5.1) to the right-hand side, over the $N(\epsilon)$ cover centers, gives: with probability at least $1-\delta$, simultaneously for all $h \in \mathcal{H}$,

$$R(h) \leq \widehat{R}_S(h) + 2G\epsilon + \sqrt{\frac{\log(2N(\epsilon)/\delta)}{2m}}.$$

**Proof.** The displayed chain of inequalities is the triangle inequality for $|\cdot|$, applied twice, with $|\widehat{R}_S(h)-\widehat{R}_S(h_j)| \leq G\epsilon$ and $|R(h)-R(h_j)| \leq G\epsilon$ from the Lipschitz property just established (the same argument bounds the empirical risk, since $\widehat{R}_S$ is the risk under the empirical distribution and the Lipschitz bound holds pointwise in $x$). The remaining term $|\widehat{R}_S(h_j)-R(h_j)|$ concerns only the $N(\epsilon)$ **fixed, data-independent** cover centers, so Theorem 5.1 applies to the finite class $\{h_1,\ldots,h_{N(\epsilon)}\}$ and bounds it, uniformly over $j$, by $\sqrt{\log(2N(\epsilon)/\delta)/(2m)}$ with probability at least $1-\delta$ $\square$

This is exactly the finite-hypothesis-class recipe of Theorem 5.1, applied not to $\mathcal{H}$ itself but to a *finite proxy* for it — the $\epsilon$-cover — with an extra $2G\epsilon$ term paying for the approximation error of using the proxy.

---

**Lemma 5a.3 (Volumetric covering bound).** Let $\mathcal{B}_B := \{v \in \mathbb{R}^d : \|v\|_2 \leq B\}$ with the Euclidean metric. Then $N(\epsilon, \mathcal{B}_B, \|\cdot\|_2) \leq \big(1 + 2B/\epsilon\big)^d$.

**Proof.** Let $\{v_1,\ldots,v_N\} \subseteq \mathcal{B}_B$ be a **maximal $\epsilon$-separated** set, i.e. $\|v_j - v_{j'}\|_2 > \epsilon$ for $j \neq j'$ and no further point of $\mathcal{B}_B$ can be added without violating this. Maximality means every $v \in \mathcal{B}_B$ is within $\epsilon$ of some $v_j$ — otherwise it could be added — so the set is an $\epsilon$-cover and $N(\epsilon,\mathcal{B}_B,\|\cdot\|_2) \leq N$.

To bound $N$, note the open balls of radius $\epsilon/2$ around the $v_j$ are pairwise disjoint (their centers are more than $\epsilon$ apart) and all contained in $\mathcal{B}_{B+\epsilon/2}$. Since the volume of a Euclidean ball of radius $r$ in $\mathbb{R}^d$ is $V_d\, r^d$ for a constant $V_d$ depending only on $d$, comparing volumes gives
$$N \cdot V_d\Big(\frac{\epsilon}{2}\Big)^d \leq V_d\Big(B + \frac{\epsilon}{2}\Big)^d \quad\Longrightarrow\quad N \leq \Big(\frac{B+\epsilon/2}{\epsilon/2}\Big)^d = \Big(1 + \frac{2B}{\epsilon}\Big)^d. \qquad\square$$

The exponent $d$ is the entire story: covering numbers grow *exponentially* in the dimension, so $\log N(\epsilon) = O(d\log(1/\epsilon))$ grows only linearly in it — and it is $\log N(\epsilon)$, not $N(\epsilon)$, that enters the bound. This is the same phenomenon as the logarithmic dependence on $|\mathcal{H}|$ in Theorem 5.1, and the reason both bounds remain useful in high dimension.

---

Balancing $\epsilon \propto 1/\sqrt m$ against the second term yields an overall rate of $O\big(\sqrt{(d/m)\log m}\big)$ — the same $\sqrt{d/m}$ rate as the VC/Rademacher bounds above, up to a $\sqrt{\log m}$ factor. That residual log factor can be removed by not fixing a single resolution $\epsilon$ but summing the bound over a geometric sequence of resolutions $\epsilon_k = 2^{-k}$ — a technique called **chaining**, which produces the sharper **Dudley entropy integral** bound $\widehat{\mathfrak{R}}_S(\mathcal{H}) \lesssim \frac{1}{\sqrt m}\int_0^\infty \sqrt{\log N(\epsilon,\mathcal{H},\mathrm{dist})}\,d\epsilon$. Chaining is beyond the scope of this course; it is mentioned here only so that the $\sqrt{\log m}$ gap between the simple covering bound above and the sharp VC-dimension rate of Theorem 5a.3 does not look like an unexplained discrepancy.

# PAC-Bayes Bounds

The bounds so far all fix a hypothesis class $\mathcal{H}$ first, and then ask for a guarantee that holds *uniformly*, for every $h \in \mathcal{H}$ at once — this is what makes it valid to then plug in whichever $h$ our learning algorithm happens to output. **PAC-Bayes bounds** [McAllester (1999)](https://doi.org/10.1145/307400.307435)<!-- cite: mcallester1999pacbayesian | inproceedings | author={McAllester, David A.}; title={PAC-Bayesian Model Averaging}; booktitle={Proceedings of the 12th Annual Conference on Computational Learning Theory (COLT)}; year={1999}; pages={164--170} --> relax this: instead of a single hypothesis, they bound the risk of a whole *distribution* over hypotheses, and let that distribution depend on the data.

**Setup.** Fix a **prior** distribution $q$ over $\mathcal{H}$, chosen *before* seeing $S$ (it may encode domain knowledge, or simply be uniform/uninformative). After seeing $S$, we are free to choose any **posterior** distribution $\rho$ over $\mathcal{H}$ — informed by the data in any way whatsoever, including via an algorithm's output — and consider the *randomized* predictor that draws a fresh $h \sim \rho$ at prediction time. Define $R(\rho) := \mathbb{E}_{h \sim \rho}[R(h)]$ and $\widehat{R}_S(\rho) := \mathbb{E}_{h\sim\rho}[\widehat{R}_S(h)]$. The bound below is governed by the **Kullback-Leibler (KL) divergence** between the two,

$$\mathrm{KL}(\rho\|q) := \mathbb{E}_{h\sim\rho}\Big[\log\frac{\rho(h)}{q(h)}\Big],$$

a measure of how distinguishable $\rho$ is from $q$: it is zero exactly when the two distributions coincide, and grows as $\rho$ places mass where $q$ places little. (Its definition is re-derived and used in the proof, where it appears as the expectation of a log-ratio; only the two displayed properties just stated are needed to read the theorem.)

**Theorem 5a.6 (PAC-Bayes bound).** Assume the loss is bounded in $[0,1]$. For any prior $q$ fixed before $S$, any $\beta>0$, and any $\delta \in (0,1)$, with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$, **simultaneously for every posterior $\rho$** (chosen after seeing $S$, in any way):

$$R(\rho) \leq \widehat{R}_S(\rho) + \frac{1}{\beta}\,\mathrm{KL}(\rho \| q) + \frac{1}{\beta}\log\frac{1}{\delta} + \frac{\beta}{8m}.$$

**Proof.** Fix $\beta>0$ and define, for each hypothesis, the random variable $X(h) := \beta\big(R(h) - \widehat{R}_S(h)\big)$, a function of $S$.

*Step 1: a moment generating function bound for a single, fixed $h$.* Write $\ell_i := \ell(y_i, h(x_i)) \in [0,1]$, independent across $i$ with mean $R(h)$. Then $X(h) = \sum_{i=1}^m \frac{\beta}{m}\big(R(h)-\ell_i\big)$ is a sum of $m$ independent, mean-zero terms, the $i$-th confined to an interval of width $\beta/m$. Hoeffding's lemma (Lemma 4.1) and independence give

$$\mathbb{E}_S\big[e^{X(h)}\big] \leq \prod_{i=1}^m \exp\Big(\frac{(\beta/m)^2}{8}\Big) = \exp\Big(\frac{\beta^2}{8m}\Big).$$

*Step 2: average over the prior, then apply Markov.* The bound of Step 1 holds for every $h$, and $q$ does not depend on $S$, so the two expectations may be exchanged:

$$\mathbb{E}_S\Big[\mathbb{E}_{h\sim q}\big[e^{X(h)}\big]\Big] = \mathbb{E}_{h\sim q}\Big[\mathbb{E}_S\big[e^{X(h)}\big]\Big] \leq e^{\beta^2/(8m)}.$$

The inner quantity $\mathbb{E}_{h\sim q}[e^{X(h)}]$ is a non-negative random variable of $S$, so Markov's inequality (Theorem 4.1) gives: with probability at least $1-\delta$ over $S$,

$$\mathbb{E}_{h\sim q}\big[e^{X(h)}\big] \;\leq\; \frac{1}{\delta}\, e^{\beta^2/(8m)}. \qquad (\ast)$$

Note carefully that $(\ast)$ involves **only the prior $q$**, so it holds on a single event of probability $1-\delta$ that is fixed before any posterior is chosen. This is the entire trick of PAC-Bayes: all the probabilistic work is done under $q$, and the data-dependent $\rho$ is introduced afterwards, deterministically, on that same event.

*Step 3: the Donsker-Varadhan change of measure.* For any distribution $\rho$ absolutely continuous with respect to $q$, and any function $X$,

$$\mathbb{E}_{h\sim\rho}\big[X(h)\big] \;\leq\; \mathrm{KL}(\rho\|q) + \log \mathbb{E}_{h\sim q}\big[e^{X(h)}\big].$$

To see this, define the **tilted** distribution $\tilde q(h) := q(h)e^{X(h)} / \mathbb{E}_{h\sim q}[e^{X(h)}]$, which is a valid probability distribution by construction. Expanding the Kullback-Leibler divergence of $\rho$ from $\tilde q$,

$$\mathrm{KL}(\rho\|\tilde q) = \mathbb{E}_{h\sim\rho}\Big[\log\frac{\rho(h)}{\tilde q(h)}\Big] = \underbrace{\mathbb{E}_{h\sim\rho}\Big[\log\frac{\rho(h)}{q(h)}\Big]}_{=\ \mathrm{KL}(\rho\|q)} - \mathbb{E}_{h\sim\rho}\big[X(h)\big] + \log\mathbb{E}_{h\sim q}\big[e^{X(h)}\big],$$

so the claimed inequality is exactly the statement $\mathrm{KL}(\rho\|\tilde q) \geq 0$, which holds for any pair of distributions by Jensen's inequality applied to the convex function $u \mapsto -\log u$.

*Step 4: combine.* On the event of $(\ast)$, for **every** $\rho$ simultaneously,

$$\beta\big(R(\rho) - \widehat{R}_S(\rho)\big) = \mathbb{E}_{h\sim\rho}\big[X(h)\big] \leq \mathrm{KL}(\rho\|q) + \log\Big(\tfrac1\delta\, e^{\beta^2/(8m)}\Big) = \mathrm{KL}(\rho\|q) + \log\tfrac1\delta + \frac{\beta^2}{8m},$$

the first equality by linearity of $\mathbb{E}_\rho$ together with the definitions $R(\rho) := \mathbb{E}_\rho[R(h)]$ and $\widehat{R}_S(\rho) := \mathbb{E}_\rho[\widehat{R}_S(h)]$. Dividing by $\beta > 0$ gives the theorem $\square$

The parameter $\beta$ — called an **inverse temperature** by analogy with statistical physics, where a high temperature flattens a distribution over states and a low one sharpens it — trades off the two error terms and can be optimized; we do this explicitly below. The minimizer of the right-hand side over $\rho$ is the **Gibbs posterior** $\widehat\rho_\beta(h) \propto \exp(-\beta\,\widehat{R}_S(h))\, q(h)$, i.e. exactly the tilted distribution $\tilde q$ of Step 3 for the choice $X = -\beta\widehat{R}_S$: the prior $q$, exponentially reweighted towards hypotheses of low empirical risk, with $\beta$ controlling how sharp the reweighting is. Note that $\widehat\rho_\beta$ coincides with the *Bayesian* posterior of Chapter 6 exactly when $\beta=m$; other choices of $\beta$ are read as different inference temperatures.

## Recovering Theorem 5.1 as a Special Case

Take $\mathcal{H} = \{h_1,\ldots,h_N\}$ finite, the **uniform prior** $q(h_j) = 1/N$, and restrict attention to **Dirac posteriors** $\rho = \delta_{h}$ that place all their mass on a single hypothesis $h \in \mathcal{H}$ — precisely the deterministic predictors we have used throughout this course. Then $R(\rho) = R(h)$, $\widehat{R}_S(\rho)=\widehat{R}_S(h)$, and $\mathrm{KL}(\delta_h \| q) = \log(1/q(h)) = \log N$, so Theorem 5a.6 gives: with probability $\geq 1-\delta$, simultaneously for **every** $h \in \mathcal{H}$,

$$\begin{gathered}
R(h) \leq \widehat{R}_S(h) + \frac{\log N}{\beta} + \frac{1}{\beta}\log\frac{1}{\delta} + \frac{\beta}{8m}, \\
\forall \beta > 0.
\end{gathered}$$

Minimizing the right-hand side over $\beta$ (setting the derivative in $\beta$ to zero gives $\beta^* = \sqrt{8m(\log N + \log(1/\delta))}$) yields exactly

$$R(h) \leq \widehat{R}_S(h) + \sqrt{\frac{\log N + \log(1/\delta)}{2m}} = \widehat{R}_S(h) + \sqrt{\frac{\log|\mathcal{H}| + \log(1/\delta)}{2m}},$$

which is **precisely Theorem 5.1**, recovered as the special case of PAC-Bayes with a uniform prior and Dirac posteriors over a finite $\mathcal{H}$. This closes the loop promised at the start of this chapter: $|\mathcal{H}|$-counting, VC dimension, Rademacher complexity, covering numbers, and PAC-Bayes bounds are not five different theories of generalization — they are one theory, instantiated through five different choices of how to measure "the complexity of $\mathcal{H}$," and each of the first four turns out to be reachable as a special case of the last, more general one.

# Beyond Complexity: What This Chapter Does Not Cover

Three further directions extend the picture above; we state them only as pointers, since a full treatment is beyond a BSc-level course.

* **Minimax lower bounds.** Every bound above is an *upper* bound: it shows some algorithm (e.g. ERM) cannot do worse than a certain rate. A **minimax lower bound** shows the converse — that *no* algorithm can do better, in the worst case over $\mathcal{D}$, than a matching rate (typically $\Omega(\sqrt{d_{VC}/m})$ under no further assumptions). Such results are proved by reducing estimation to hypothesis testing over a carefully chosen finite set of hard distributions, then invoking information-theoretic tools (Fano's inequality or its relatives). They certify that the rates derived in this chapter are not an artifact of a loose proof technique, but essentially unimprovable.
* **Margin theory and fast rates.** All bounds above scale as $O(1/\sqrt m)$. Under a **low-noise (margin) condition** — informally, when $\mathcal{D}$ rarely places points close to the decision boundary — the *same* complexity measures support sharper $O(1/m)$ "fast rates," interpolating continuously between the two regimes as the margin condition strengthens. This refines, rather than replaces, the bias-complexity dilemma of Chapter 5: near the decision boundary is exactly where estimation error concentrates, so a distribution that avoids that region is intrinsically easier to learn, beyond what its VC dimension alone would suggest.
* **Algorithmic stability.** Every bound in this chapter is a property of the hypothesis *class* $\mathcal{H}$, independent of which algorithm searches it. A complementary theory bounds generalization instead via the **stability** of the specific learning algorithm — informally, how much its output hypothesis can change if a single training point is swapped out — and can certify generalization even in settings (e.g. certain non-ERM algorithms, or classes with poorly behaved uniform convergence) where the complexity-based bounds of this chapter are vacuous or inapplicable. It is the standard tool, for instance, for analyzing the generalization behavior of stochastic gradient descent itself (Chapter 8), independently of the capacity of the neural network it is training.
