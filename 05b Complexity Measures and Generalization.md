Chapter 6 built the entire theory of generalization on top of two concrete complexity measures: the raw count $|\mathcal{H}|$ for finite classes, and the VC dimension $d_{VC}(\mathcal{H})$ for infinite ones. Both did the same job: they compressed a hypothesis class down to a single number that a Hoeffding-type concentration inequality, combined with the union bound, could feed on. This chapter makes that role explicit and abstract, then instantiates it three more times.

**The general form.** Every bound in this course, past and future, has the shape

$$\begin{gathered}
R(h) \leq \widehat{R}_S(h) + \underbrace{\mathrm{Complexity}(\mathcal{H}, m)}_{\text{shrinks as } m \to \infty} + \underbrace{\mathrm{Confidence}(\delta, m)}_{\text{shrinks as } m \to \infty}, \\
\forall h \in \mathcal{H}, \text{ w.p.} \geq 1-\delta,
\end{gathered}$$

i.e. a bound on the **representativeness** gap $\sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ from Definition 6.2.2. What is negotiable is only how $\mathrm{Complexity}(\mathcal{H},m)$ is defined. We already have two instances: $\mathrm{Complexity} = \sqrt{\log|\mathcal{H}|/(2m)}$ (Theorem 6.1.3) and $\mathrm{Complexity} = \sqrt{d_{VC}(\mathcal{H})/m}$ up to logarithmic factors (Theorem 6.4.7). We now add three more: **Rademacher complexity**, **covering numbers**, and **PAC-Bayes bounds**. All five are, in a precise sense we make explicit below, the same quantity viewed through different lenses — and by the end of this chapter, we will have shown algebraically that the very first bound of Chapter 6 (Theorem 6.1.3) is a special case of the PAC-Bayes bound. The second half of the chapter then sharpens the VC rates by chaining, prices norm-constrained combinations, introduces algorithmic stability, and shows by minimax lower bounds that the rates obtained cannot be improved in general.

# A Concentration Tool for Suprema: McDiarmid's Inequality

Hoeffding's inequality (Chapter 5) concentrates a *single* average $\widehat\mu_m$ around its mean. Bounding $\sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ requires concentrating a *function of the whole sample* — the supremum over (possibly infinitely many) hypotheses — around its mean. The right tool is a generalization, due to [McDiarmid (1989)](https://doi.org/10.1017/CBO9781107359949.008)<!-- cite: mcdiarmid1989method | article | author={McDiarmid, Colin}; title={On the Method of Bounded Differences}; journal={Surveys in Combinatorics}; year={1989}; volume={141}; pages={148--188} -->, of Hoeffding's inequality to arbitrary functions of independent random variables, provided the function does not depend too sensitively on any single one of them.

**Theorem 7.1.1 (McDiarmid's bounded differences inequality).** Let $Z_1,\ldots,Z_m$ be independent random variables and let $f: \mathcal{Z}^m \to \mathbb{R}$ satisfy the **bounded differences** property: for every $i \in [m]$ and every $z_1,\ldots,z_m,z_i' \in \mathcal{Z}$,

$$\big| f(z_1,\ldots,z_i,\ldots,z_m) - f(z_1,\ldots,z_i',\ldots,z_m) \big| \leq c_i.$$

Then for any $t > 0$,

$$P\big(|f(Z_1,\ldots,Z_m) - \mathbb{E}[f(Z_1,\ldots,Z_m)]| \geq t\big) \leq 2\exp\Big(\frac{-2t^2}{\sum_{i=1}^m c_i^2}\Big).$$

**Proof.** The argument is the Chernoff route of Theorem 5.6.5, applied not to the summands themselves — there are none — but to an artificial decomposition of $f$ into $m$ increments, one per input variable. Write $Z_{1:i} := (Z_1,\ldots,Z_i)$, $F := f(Z_1,\ldots,Z_m)$, and define the **increments**

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

*Step 2: conditional Hoeffding's lemma, iterated.* Lemma 5.6.4 applies verbatim to conditional expectations, so for every $s \in \mathbb{R}$,
$$\mathbb{E}\big[e^{sV_i} \mid Z_{1:i-1}\big] \leq e^{s^2c_i^2/8}.$$
Peeling the increments off one at a time with the tower property,
$$\mathbb{E}\big[e^{s\sum_{i=1}^m V_i}\big] = \mathbb{E}\Big[e^{s\sum_{i<m}V_i}\ \underbrace{\mathbb{E}\big[e^{sV_m}\mid Z_{1:m-1}\big]}_{\leq\ e^{s^2c_m^2/8}}\Big] \leq \cdots \leq \exp\Big(\frac{s^2}{8}\sum_{i=1}^m c_i^2\Big).$$

*Step 3: Chernoff.* Exactly as in Theorem 5.6.5, Markov's inequality on $e^{s(F-\mathbb{E}[F])}$ gives $P(F - \mathbb{E}[F] \geq t) \leq \exp(-st + \tfrac{s^2}{8}\sum_i c_i^2)$ for every $s>0$; minimizing the exponent at $s^* = 4t/\sum_i c_i^2$ yields $\exp\big(-2t^2/\sum_i c_i^2\big)$. Applying the same argument to $-f$ bounds the other tail identically, and adding the two gives the factor $2$ $\square$

Setting $m=1$, or taking $f(z_1,\ldots,z_m) = \frac1m\sum_i z_i$ with $c_i = (b_i-a_i)/m$, recovers Hoeffding's inequality — McDiarmid's inequality is its strict generalization, obtained by replacing "sum of independent variables" with "any function that no single variable can move very far".

For the choice $f(S) := \sup_{h \in \mathcal{H}} \big(R(h) - \widehat{R}_S(h)\big)$ with a loss bounded in $[0,1]$, changing a single training example changes $\widehat{R}_S(h)$ by at most $1/m$ for every $h$ simultaneously, hence $f$ has bounded differences $c_i = 1/m$, and McDiarmid's inequality applies with $\sum_i c_i^2 = 1/m$. This is exactly the role Hoeffding's inequality played for a single hypothesis in Chapter 5 — McDiarmid is its supremum-over-a-class counterpart, and it is what turns a bound on $\mathbb{E}_S[\sup_h(R(h)-\widehat{R}_S(h))]$ into the high-probability statements below.

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

**Definition 7.3.1 (Rademacher complexity).** For a set $A \subset \mathbb{R}^m$ and i.i.d. Rademacher signs $\sigma_1,\ldots,\sigma_m \in \{-1,+1\}$ ($P(\sigma_i=1)=P(\sigma_i=-1)=\tfrac12$), define

$$\widehat{\mathfrak{R}}(A) := \mathbb{E}_\sigma\Big[\sup_{a \in A} \frac{1}{m}\sum_{i=1}^m \sigma_i a_i\Big].$$

For a hypothesis class $\mathcal{H}$ and a fixed sample $S = \{(x_i,y_i)\}_{i=1}^m$, write $\ell \circ \mathcal{H} \circ S := \big\{(\ell(y_1, h(x_1)),\ldots,\ell(y_m, h(x_m))) : h \in \mathcal{H}\big\} \subset \mathbb{R}^m$ for the loss class evaluated on $S$, and define the **empirical Rademacher complexity** $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) := \widehat{\mathfrak{R}}(\ell\circ\mathcal{H}\circ S)$ — a data-dependent quantity, computable from $S$ alone — and the **(expected) Rademacher complexity** $\mathfrak{R}_m(\ell\circ\mathcal{H}) := \mathbb{E}_S\big[\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})\big]$. When we want the complexity of the hypothesis class itself rather than of the loss class, we write $\widehat{\mathfrak{R}}_S(\mathcal{H}) := \widehat{\mathfrak{R}}(\mathcal{H}_S)$, evaluated on the restriction $\mathcal{H}_S$ of Definition 6.4.1.

---

Intuitively, $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$ measures how well $\mathcal{H}$ can fit **pure random noise**: $\sigma_i$ is a coin flip drawn independently of the data, so $\sup_h \frac1m\sum_i \sigma_i \ell(y_i, h(x_i))$ asks how large a spurious correlation the richest hypothesis in $\mathcal{H}$ can manufacture with random labels. A class rich enough to realize any loss vector in $\{0,1\}^m$ (as in the memorization hypothesis of Chapter 2) can match every sign, giving $\widehat{\mathfrak{R}}_S \approx \tfrac12$. A class with a single hypothesis has no supremum to exploit, so $\widehat{\mathfrak{R}}_S = \frac{1}{m}\sum_i \mathbb{E}[\sigma_i]\,a_i = 0$ exactly.

**Theorem 7.3.2 (Rademacher generalization bound).** Let the loss be bounded in $[0,1]$. For any $\delta \in (0,1)$, with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$, simultaneously for all $h \in \mathcal{H}$:

$$R(h) \leq \widehat{R}_S(h) + 2\,\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + 3\sqrt{\frac{\log(2/\delta)}{2m}}.$$

**Proof.** Write $\Phi(S) := \sup_{h\in\mathcal{H}}\big(R(h)-\widehat{R}_S(h)\big)$, so the claim is a high-probability upper bound on $\Phi(S)$. The proof applies McDiarmid's inequality twice, with symmetrization in between.

*Step 1: concentrate $\Phi$ around its mean.* Replacing one example of $S$ changes $\widehat{R}_S(h)$ by at most $1/m$ for every $h$ at once, hence changes the supremum by at most $1/m$: $\Phi$ has bounded differences $c_i = 1/m$, so $\sum_i c_i^2 = 1/m$. The one-sided form of Theorem 7.1.1 at confidence $\delta/2$ gives, with probability at least $1-\delta/2$,
$$\Phi(S) \;\leq\; \mathbb{E}_S[\Phi(S)] + \sqrt{\frac{\log(2/\delta)}{2m}}.$$

*Step 2: bound the mean by the expected Rademacher complexity.* This is the symmetrization argument of the previous section, verbatim: $\mathbb{E}_S[\Phi(S)] \leq 2\,\mathfrak{R}_m(\ell\circ\mathcal{H})$.

*Step 3: replace the expected complexity by its empirical counterpart.* The map $S \mapsto \widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$ also has bounded differences $1/m$: changing one example changes each coordinate of the loss vector by at most $1$, hence changes $\frac1m\sum_i\sigma_i a_i$ by at most $1/m$ uniformly, and a supremum of functions that all move by at most $1/m$ moves by at most $1/m$. Applying Theorem 7.1.1 once more at confidence $\delta/2$, with probability at least $1-\delta/2$,
$$\mathfrak{R}_m(\ell\circ\mathcal{H}) = \mathbb{E}_S\big[\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})\big] \;\leq\; \widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + \sqrt{\frac{\log(2/\delta)}{2m}}.$$

*Step 4: union bound.* Both events hold simultaneously with probability at least $1-\delta$, and on their intersection
$$\sup_{h}\big(R(h)-\widehat{R}_S(h)\big) \leq 2\,\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) + 2\sqrt{\frac{\log(2/\delta)}{2m}} + \sqrt{\frac{\log(2/\delta)}{2m}},$$
where the factor $2$ from Step 2 multiplies the deviation of Step 3. Rearranging the supremum gives the claim for every $h$ $\square$

This is the Rademacher-complexity instance of the general form stated at the top of this chapter, with $\mathrm{Complexity}(\mathcal{H},m) = 2\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H})$.

## Massart's Lemma, and Theorem 6.1.3 revisited

**Lemma 7.3.3 (Massart's lemma).** For a finite set $A = \{a^{(1)},\ldots,a^{(N)}\} \subset \mathbb{R}^m$ with $\|a^{(j)}\|_2 \leq B$ for all $j$,

$$\begin{gathered}
\widehat{\mathfrak{R}}(A) \leq \frac{B\sqrt{2\log N}}{m}, \\
\text{equivalently} \quad \mathbb{E}_\sigma\Big[\max_{j\in[N]}\langle \sigma, a^{(j)}\rangle\Big] \leq B\sqrt{2\log N}.
\end{gathered}$$

**Proof.** Write $X_j := \langle\sigma, a^{(j)}\rangle = \sum_{i=1}^m \sigma_i a_i^{(j)}$ and assume $N \geq 2$ (for $N=1$ the left side is $\mathbb{E}_\sigma[X_1] = 0$ and there is nothing to prove). Fix $s>0$. Because $u \mapsto e^{su}$ is convex and increasing, Jensen's inequality and then the crude bound "a maximum of non-negative terms is at most their sum" give

$$\exp\Big(s\,\mathbb{E}_\sigma\big[\max_j X_j\big]\Big) \;\leq\; \mathbb{E}_\sigma\Big[e^{s \max_j X_j}\Big] = \mathbb{E}_\sigma\Big[\max_j e^{s X_j}\Big] \;\leq\; \sum_{j=1}^N \mathbb{E}_\sigma\big[e^{sX_j}\big].$$

Each $X_j$ is a sum of $m$ independent, mean-zero terms $\sigma_i a_i^{(j)}$, the $i$-th confined to the interval $[-|a_i^{(j)}|, +|a_i^{(j)}|]$ of width $2|a_i^{(j)}|$. Hoeffding's lemma (Lemma 5.6.4) applied to each term and independence give

$$\mathbb{E}_\sigma\big[e^{sX_j}\big] = \prod_{i=1}^m \mathbb{E}\big[e^{s\sigma_i a_i^{(j)}}\big] \leq \prod_{i=1}^m \exp\Big(\frac{s^2 (2a_i^{(j)})^2}{8}\Big) = \exp\Big(\frac{s^2\|a^{(j)}\|_2^2}{2}\Big) \leq \exp\Big(\frac{s^2B^2}{2}\Big).$$

Substituting and taking logarithms, $s\,\mathbb{E}_\sigma[\max_j X_j] \leq \log N + \tfrac{s^2B^2}{2}$, i.e.

$$\begin{gathered}
\mathbb{E}_\sigma\big[\max_j X_j\big] \leq \frac{\log N}{s} + \frac{sB^2}{2} \\
\text{for every } s>0.
\end{gathered}$$

The right-hand side is minimized at $s^* = \sqrt{2\log N}/B$, where its value is $B\sqrt{2\log N}$. Dividing by $m$ turns the left-hand side into $\widehat{\mathfrak{R}}(A)$ $\square$

Note the shape of the result: the dependence on the number of elements is only $\sqrt{\log N}$, which is what makes the union-bound arguments of Chapter 6 so efficient. Note also that the bound is *not* improved by adding a point to $A$ that is far from the others — only the count and the worst-case norm matter.

Apply this to $\ell\circ\mathcal{H}\circ S$ for a **finite** hypothesis class, $N = |\mathcal{H}|$, with loss bounded in $[0,1]$ so $\|a^{(j)}\|_2 \leq \sqrt m$: this gives $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) \leq \sqrt{2\log|\mathcal{H}|/m}$, and plugging into Theorem 7.3.2 recovers, up to constants, exactly the finite-hypothesis-class bound of Theorem 6.1.3. **The very first generalization bound of Chapter 6 was already an instance of a Rademacher-complexity bound** — we simply proved it directly via Hoeffding and the union bound there, without naming the general machinery.

## The Contraction Lemma

**Lemma 7.3.4 (Contraction).** If $\phi_i: \mathbb{R} \to \mathbb{R}$ is $\rho$-Lipschitz for each $i$, then $\widehat{\mathfrak{R}}\big(\{(\phi_1(a_1),\ldots,\phi_m(a_m)) : a \in A\}\big) \leq \rho\, \widehat{\mathfrak{R}}(A)$.

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

This lets us pass, in the definition of $\ell \circ \mathcal{H}$, from the Rademacher complexity of the *loss-composed* class directly to the Rademacher complexity of $\mathcal{H}$ itself, whenever the loss $\ell(\cdot,y)$ is $\rho$-Lipschitz in its first argument: $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) \leq \rho\,\widehat{\mathfrak{R}}_S(\mathcal{H})$. A word of caution, since this is a place where different textbooks state slightly different constants: the *clean* version above, with no extra factor, holds for the **unsigned** Rademacher complexity (where the supremum in Definition 7.3.1 is *not* wrapped in an absolute value, exactly as we defined it). If instead one works with the signed variant $\sup_a |\frac1m\sum_i\sigma_i a_i|$, the contraction inequality picks up an extra factor of $2$. Precisely: if $0 \in A$ and $\phi_i(0) = 0$ for all $i$, write $\phi(A) := \{(\phi_1(a_1),\ldots,\phi_m(a_m)) : a \in A\}$, which then contains $0$. For every realization of the signs both $\sup_{b\in\phi(A)}\langle\sigma,b\rangle$ and $\sup_{b\in\phi(A)}\langle-\sigma,b\rangle$ are $\geq 0$ (take $b = 0$), so
$$\sup_{b\in\phi(A)}\big|\langle\sigma,b\rangle\big| = \max\Big(\sup_{b}\langle\sigma,b\rangle,\ \sup_{b}\langle-\sigma,b\rangle\Big) \leq \sup_{b}\langle\sigma,b\rangle + \sup_{b}\langle-\sigma,b\rangle.$$
The two terms have the same expectation, since $-\sigma$ has the same law as $\sigma$, and the lemma bounds each by $\rho\,m\,\widehat{\mathfrak{R}}(A)$. Hence
$$\mathbb{E}_\sigma\Big[\sup_{a\in A}\Big|\frac1m\sum_i\sigma_i\phi_i(a_i)\Big|\Big] \;\leq\; 2\rho\,\widehat{\mathfrak{R}}(A). \tag{7.1}$$
Since our Definition 7.3.1 uses the unsigned form throughout, the clean statement of the lemma applies directly throughout this chapter; (7.1) is needed only in Chapter 10.

## The Bridge to VC Dimension

**Theorem 7.3.5 (VC dimension bounds Rademacher complexity).** For a hypothesis class $\mathcal{H}$ of binary classifiers with $d_{VC}(\mathcal{H}) = d < \infty$, and the zero-one loss,

$$\begin{gathered}
\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq 2\sqrt{\frac{d \log(m+1)}{m}}, \\
\text{for every sample } S \text{ of size } m.
\end{gathered}$$

**Proof sketch.** The restriction $\mathcal{H}_S = \{(h(x_1),\ldots,h(x_m)) : h \in \mathcal{H}\}$ is a *finite* set (Definition 6.4.1), of size $|\mathcal{H}_S| = \tau_{\mathcal{H}}(m) \leq (m+1)^d$ by the Sauer-Shelah-Perles lemma (Lemma 6.4.8, using the simplified polynomial form). Apply Massart's lemma (Lemma 7.3.3) with $N = (m+1)^d$ and $B=\sqrt m$ (binary values in $\{0,1\}$, so $\|a^{(j)}\|_2 \le \sqrt{m}$):

$$\widehat{\mathfrak{R}}(\mathcal{H}_S) \leq \frac{\sqrt{m}\sqrt{2\log\big((m+1)^d\big)}}{m} = \sqrt{\frac{2 d\log(m+1)}{m}} \leq 2\sqrt{\frac{d\log(m+1)}{m}},$$

the last step simply using $\sqrt 2 \leq 2$, which is where the slack in the stated constant comes from. $\square$

This is the promised unification made concrete: **VC dimension is not a separate theory from Rademacher complexity — it is a specific, combinatorial way of bounding it**, via Sauer-Shelah-Perles turning an infinite hypothesis class into a finite restriction on any given sample, to which Massart's finite-class bound then applies.

## A VC-Dimension-Free Example: Linear Predictors

One reason Rademacher complexity is worth the extra abstraction: it can be **finite and small even when $d_{VC}(\mathcal{H}) = \infty$**, provided we constrain the hypothesis class by a *norm* rather than by dimension.

**Theorem 7.3.6.** Let $\mathcal{H} = \{x \mapsto w^\top x : \|w\|_2 \leq B\}$ and suppose $\|x\|_2 \leq R$ almost surely under $\mathcal{D}$. Then

$$\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq \frac{BR}{\sqrt{m}}.$$

**Proof.** By definition and linearity, for a fixed realization of the signs,
$$\sup_{\|w\|_2\leq B} \frac1m \sum_{i=1}^m \sigma_i\, w^\top x_i = \frac1m \sup_{\|w\|_2 \leq B} \Big\langle w, \sum_{i=1}^m \sigma_i x_i\Big\rangle = \frac{B}{m}\Big\|\sum_{i=1}^m \sigma_i x_i\Big\|_2,$$
where the second equality is the equality case of the Cauchy-Schwarz inequality: the linear functional $w \mapsto \langle w, v\rangle$ is maximized over the ball $\|w\|_2\leq B$ at $w = B\,v/\|v\|_2$, with value $B\|v\|_2$. Taking $\mathbb{E}_\sigma$ and applying Jensen's inequality to the concave map $u \mapsto \sqrt u$,
$$\widehat{\mathfrak{R}}_S(\mathcal{H}) = \frac{B}{m}\,\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2\Big] \leq \frac{B}{m}\sqrt{\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2^2\Big]}.$$
Expanding the squared norm and using $\mathbb{E}[\sigma_i\sigma_{i'}] = \mathds{1}(i=i')$ (the signs are independent and mean-zero, and $\sigma_i^2=1$),
$$\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i x_i\Big\|_2^2\Big] = \sum_{i,i'} \mathbb{E}[\sigma_i\sigma_{i'}]\, x_i^\top x_{i'} = \sum_{i=1}^m \|x_i\|_2^2 \leq mR^2 .$$
Combining, $\widehat{\mathfrak{R}}_S(\mathcal{H}) \leq \frac{B}{m}\sqrt{mR^2} = \frac{BR}{\sqrt m}$ $\square$

Notice this bound does not depend on the input dimension $d$ at all — only on the norm bound $B$ and the data norm bound $R$. This matters directly for the kernel methods of Chapter 11, where $\phi(x)$ can live in an infinite-dimensional feature space (for which $d_{VC}$ is typically infinite) while $\|w\|_2$ and $\|\phi(x)\|_2$ remain finite: Rademacher complexity, unlike VC dimension, gives a meaningful, finite generalization guarantee in exactly this regime.

# Covering Numbers

Rademacher complexity measures $\mathcal{H}$'s capacity to correlate with noise. **Covering numbers** measure a more concrete quantity: how many hypotheses it takes to approximate every hypothesis in $\mathcal{H}$ to a given tolerance.

**Definition 7.4.1 ($\epsilon$-cover, covering number).** Let $\mathcal{H}$ be a hypothesis class equipped with a (pseudo-)metric $\mathrm{dist}(\cdot,\cdot)$: a function satisfying axioms M2 and M3 of the metric-space definition of Chapter 3, while M1 is relaxed to the one-directional $\mathrm{dist}(h,h') = 0 \Leftarrow h = h'$, so that distinct hypotheses may sit at distance zero. A finite set $\{h_1,\ldots,h_N\} \subset \mathcal{H}$ is an **$\epsilon$-cover** of $\mathcal{H}$ if for every $h \in \mathcal{H}$ there exists $j \in [N]$ with $\mathrm{dist}(h,h_j) \leq \epsilon$. The **covering number** $N(\epsilon,\mathcal{H},\mathrm{dist})$ is the size of the smallest such cover.

---

In words, an $\epsilon$-cover is a finite "grid" of representative hypotheses such that every hypothesis in $\mathcal{H}$, however large or infinite the class, is within $\epsilon$ of some grid point — turning an infinite class into a finite proxy at the cost of an $\epsilon$-sized approximation error.

The relevant metric for generalization is one under which the risk is Lipschitz: if $\ell(\cdot,y)$ is $G$-Lipschitz in its first argument for every $y$, and $\mathrm{dist}(h,h') := \mathbb{E}_{x\sim\mathcal{D}}|h(x)-h'(x)|$, then $|R(h)-R(h')| \leq G\,\mathrm{dist}(h,h')$ for all $h,h' \in \mathcal{H}$ — replacing $h$ by the nearest cover element changes the risk by only $G\epsilon$. This particular $\mathrm{dist}$ is a genuine pseudo-metric rather than a metric: two distinct hypotheses that differ only on inputs that $\mathcal{D}$ never draws have distance zero, which is exactly why M1 had to be relaxed above — and it is harmless, since such hypotheses also have identical risk.

**Theorem 7.4.2 (Covering-number generalization bound).** Let $\{h_1,\ldots,h_{N(\epsilon)}\}$ be an $\epsilon$-cover of $\mathcal{H}$ under $\mathrm{dist}$ as above, with loss bounded in $[0,1]$. Then, for any $h \in \mathcal{H}$, choosing $h_j$ within $\epsilon$ of $h$,

$$
\begin{aligned}
\big|\widehat{R}_S(h) - R(h)\big| &\leq \big|\widehat{R}_S(h) - \widehat{R}_S(h_j)\big| + \big|\widehat{R}_S(h_j)-R(h_j)\big| + \big|R(h_j)-R(h)\big| \\
&\leq 2G\epsilon + \big|\widehat{R}_S(h_j)-R(h_j)\big|,
\end{aligned}
$$

using the Lipschitz bound on both the empirical and true risk. Applying the finite-hypothesis-class bound (Theorem 6.1.3) to the right-hand side, over the $N(\epsilon)$ cover centers, gives: with probability at least $1-\delta$, simultaneously for all $h \in \mathcal{H}$,

$$R(h) \leq \widehat{R}_S(h) + 2G\epsilon + \sqrt{\frac{\log(2N(\epsilon)/\delta)}{2m}}.$$

**Proof.** The displayed chain of inequalities is the triangle inequality for $|\cdot|$, applied twice, with $|\widehat{R}_S(h)-\widehat{R}_S(h_j)| \leq G\epsilon$ and $|R(h)-R(h_j)| \leq G\epsilon$ from the Lipschitz property just established (the same argument bounds the empirical risk, since $\widehat{R}_S$ is the risk under the empirical distribution and the Lipschitz bound holds pointwise in $x$). The remaining term $|\widehat{R}_S(h_j)-R(h_j)|$ concerns only the $N(\epsilon)$ **fixed, data-independent** cover centers, so Theorem 6.1.3 applies to the finite class $\{h_1,\ldots,h_{N(\epsilon)}\}$ and bounds it, uniformly over $j$, by $\sqrt{\log(2N(\epsilon)/\delta)/(2m)}$ with probability at least $1-\delta$ $\square$

This is exactly the finite-hypothesis-class argument of Theorem 6.1.3, applied not to $\mathcal{H}$ itself but to a *finite proxy* for it — the $\epsilon$-cover — with an extra $2G\epsilon$ term paying for the approximation error of using the proxy.

---

**Lemma 7.4.3 (Volumetric covering bound).** Let $\mathcal{B}_B := \{v \in \mathbb{R}^d : \|v\|_2 \leq B\}$ with the Euclidean metric. Then $N(\epsilon, \mathcal{B}_B, \|\cdot\|_2) \leq \big(1 + 2B/\epsilon\big)^d$.

**Proof.** Let $\{v_1,\ldots,v_N\} \subseteq \mathcal{B}_B$ be a **maximal $\epsilon$-separated** set, i.e. $\|v_j - v_{j'}\|_2 > \epsilon$ for $j \neq j'$ and no further point of $\mathcal{B}_B$ can be added without violating this. Maximality means every $v \in \mathcal{B}_B$ is within $\epsilon$ of some $v_j$ — otherwise it could be added — so the set is an $\epsilon$-cover and $N(\epsilon,\mathcal{B}_B,\|\cdot\|_2) \leq N$.

To bound $N$, note the open balls of radius $\epsilon/2$ around the $v_j$ are pairwise disjoint (their centers are more than $\epsilon$ apart) and all contained in $\mathcal{B}_{B+\epsilon/2}$. Since the volume of a Euclidean ball of radius $r$ in $\mathbb{R}^d$ is $V_d\, r^d$ for a constant $V_d$ depending only on $d$, comparing volumes gives
$$N \cdot V_d\Big(\frac{\epsilon}{2}\Big)^d \leq V_d\Big(B + \frac{\epsilon}{2}\Big)^d \quad\Longrightarrow\quad N \leq \Big(\frac{B+\epsilon/2}{\epsilon/2}\Big)^d = \Big(1 + \frac{2B}{\epsilon}\Big)^d. \qquad\square$$

The exponent $d$ is the entire story: covering numbers grow *exponentially* in the dimension, so $\log N(\epsilon) = O(d\log(1/\epsilon))$ grows only linearly in it — and it is $\log N(\epsilon)$, not $N(\epsilon)$, that enters the bound. This is the same phenomenon as the logarithmic dependence on $|\mathcal{H}|$ in Theorem 6.1.3, and the reason both bounds remain useful in high dimension.

---

Balancing $\epsilon \propto 1/\sqrt m$ against the second term yields an overall rate of $O\big(\sqrt{(d/m)\log m}\big)$ — the same $\sqrt{d/m}$ rate as the VC/Rademacher bounds above, up to a $\sqrt{\log m}$ factor. That residual log factor can be removed by not fixing a single resolution $\epsilon$ but summing the bound over a geometric sequence of resolutions $\epsilon_k = 2^{-k}$ — a technique called **chaining**. We return to it after the PAC-Bayes bounds, where it removes this factor for VC classes (Corollary 7.6.3).

# PAC-Bayes Bounds

The bounds so far all fix a hypothesis class $\mathcal{H}$ first, and then ask for a guarantee that holds *uniformly*, for every $h \in \mathcal{H}$ at once — this is what makes it valid to then plug in whichever $h$ our learning algorithm happens to output. **PAC-Bayes bounds** [McAllester (1999)](https://doi.org/10.1145/307400.307435)<!-- cite: mcallester1999pacbayesian | inproceedings | author={McAllester, David A.}; title={PAC-Bayesian Model Averaging}; booktitle={Proceedings of the 12th Annual Conference on Computational Learning Theory (COLT)}; year={1999}; pages={164--170} --> relax this: instead of a single hypothesis, they bound the risk of a whole *distribution* over hypotheses, and let that distribution depend on the data.

**Setup.** Fix a **prior** distribution $q$ over $\mathcal{H}$, chosen *before* seeing $S$ (it may encode domain knowledge, or simply be uniform/uninformative). After seeing $S$, we are free to choose any **posterior** distribution $\rho$ over $\mathcal{H}$ — informed by the data in any way whatsoever, including via an algorithm's output — and consider the *randomized* predictor that draws a fresh $h \sim \rho$ at prediction time. Define $R(\rho) := \mathbb{E}_{h \sim \rho}[R(h)]$ and $\widehat{R}_S(\rho) := \mathbb{E}_{h\sim\rho}[\widehat{R}_S(h)]$. The bound below is governed by the **Kullback-Leibler (KL) divergence** between the two,

$$\mathrm{KL}(\rho\|q) := \mathbb{E}_{h\sim\rho}\Big[\log\frac{\rho(h)}{q(h)}\Big],$$

a measure of how distinguishable $\rho$ is from $q$: it is zero exactly when the two distributions coincide, and grows as $\rho$ places mass where $q$ places little. (Its definition is re-derived and used in the proof, where it appears as the expectation of a log-ratio; only the two displayed properties just stated are needed to read the theorem.)

**Theorem 7.5.1 (PAC-Bayes bound).** Assume the loss is bounded in $[0,1]$. For any prior $q$ fixed before $S$, any $\beta>0$, and any $\delta \in (0,1)$, with probability at least $1-\delta$ over $S \sim \mathcal{D}^m$, **simultaneously for every posterior $\rho$** (chosen after seeing $S$, in any way):

$$R(\rho) \leq \widehat{R}_S(\rho) + \frac{1}{\beta}\,\mathrm{KL}(\rho \| q) + \frac{1}{\beta}\log\frac{1}{\delta} + \frac{\beta}{8m}.$$

**Proof.** Fix $\beta>0$ and define, for each hypothesis, the random variable $X(h) := \beta\big(R(h) - \widehat{R}_S(h)\big)$, a function of $S$.

*Step 1: a moment generating function bound for a single, fixed $h$.* Write $\ell_i := \ell(y_i, h(x_i)) \in [0,1]$, independent across $i$ with mean $R(h)$. Then $X(h) = \sum_{i=1}^m \frac{\beta}{m}\big(R(h)-\ell_i\big)$ is a sum of $m$ independent, mean-zero terms, the $i$-th confined to an interval of width $\beta/m$. Hoeffding's lemma (Lemma 5.6.4) and independence give

$$\mathbb{E}_S\big[e^{X(h)}\big] \leq \prod_{i=1}^m \exp\Big(\frac{(\beta/m)^2}{8}\Big) = \exp\Big(\frac{\beta^2}{8m}\Big).$$

*Step 2: average over the prior, then apply Markov.* The bound of Step 1 holds for every $h$, and $q$ does not depend on $S$, so the two expectations may be exchanged:

$$\mathbb{E}_S\Big[\mathbb{E}_{h\sim q}\big[e^{X(h)}\big]\Big] = \mathbb{E}_{h\sim q}\Big[\mathbb{E}_S\big[e^{X(h)}\big]\Big] \leq e^{\beta^2/(8m)}.$$

The inner quantity $\mathbb{E}_{h\sim q}[e^{X(h)}]$ is a non-negative random variable of $S$, so Markov's inequality (Theorem 5.6.1) gives: with probability at least $1-\delta$ over $S$,

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

The parameter $\beta$ — called an **inverse temperature** by analogy with statistical physics, where a high temperature flattens a distribution over states and a low one sharpens it — trades off the two error terms and can be optimized; we do this explicitly below. The minimizer of the right-hand side over $\rho$ is the **Gibbs posterior** $\widehat\rho_\beta(h) \propto \exp(-\beta\,\widehat{R}_S(h))\, q(h)$, i.e. exactly the tilted distribution $\tilde q$ of Step 3 for the choice $X = -\beta\widehat{R}_S$: the prior $q$, exponentially reweighted towards hypotheses of low empirical risk, with $\beta$ controlling how sharp the reweighting is. Note that $\widehat\rho_\beta$ coincides with the *Bayesian* posterior of Chapter 8 exactly when $\beta=m$; other choices of $\beta$ are read as different inference temperatures.

## Recovering Theorem 6.1.3 as a Special Case

Take $\mathcal{H} = \{h_1,\ldots,h_N\}$ finite, the **uniform prior** $q(h_j) = 1/N$, and restrict attention to **Dirac posteriors** $\rho = \delta_{h}$ that place all their mass on a single hypothesis $h \in \mathcal{H}$ — precisely the deterministic predictors we have used throughout this course. Then $R(\rho) = R(h)$, $\widehat{R}_S(\rho)=\widehat{R}_S(h)$, and $\mathrm{KL}(\delta_h \| q) = \log(1/q(h)) = \log N$, so Theorem 7.5.1 gives: with probability $\geq 1-\delta$, simultaneously for **every** $h \in \mathcal{H}$,

$$\begin{gathered}
R(h) \leq \widehat{R}_S(h) + \frac{\log N}{\beta} + \frac{1}{\beta}\log\frac{1}{\delta} + \frac{\beta}{8m}, \\
\forall \beta > 0.
\end{gathered}$$

Minimizing the right-hand side over $\beta$ (setting the derivative in $\beta$ to zero gives $\beta^* = \sqrt{8m(\log N + \log(1/\delta))}$) yields exactly

$$R(h) \leq \widehat{R}_S(h) + \sqrt{\frac{\log N + \log(1/\delta)}{2m}} = \widehat{R}_S(h) + \sqrt{\frac{\log|\mathcal{H}| + \log(1/\delta)}{2m}},$$

which is **precisely Theorem 6.1.3**, recovered as the special case of PAC-Bayes with a uniform prior and Dirac posteriors over a finite $\mathcal{H}$. This closes the loop promised at the start of this chapter: $|\mathcal{H}|$-counting, VC dimension, Rademacher complexity, covering numbers, and PAC-Bayes bounds are not five different theories of generalization — they are one theory, instantiated through five different choices of how to measure "the complexity of $\mathcal{H}$," and each of the first four turns out to be reachable as a special case of the last, more general one.

# Chaining: Sharp Rates for VC Classes

The five complexity measures above complete the abstract framework. The remaining sections refine it in four directions: chaining removes the logarithmic factor left over in the VC bounds (this section); $L_1$-constrained combinations show that capacity can be priced by a norm on the combination weights (next section); algorithmic stability prices the *algorithm* rather than the class; and minimax lower bounds show that the rates obtained cannot be improved in general.

Recall the residual $\sqrt{\log m}$ factor of the single-resolution covering bound, and the $\sqrt{\log(m+1)}$ factor of Theorem 7.3.5. Both are removed by covering the class at all resolutions $\epsilon_k = 2^{-k}$ simultaneously. Two ingredients are needed: a covering-number estimate for VC classes that does not depend on the sample, and the chaining bound itself. Throughout, for a sample $S = \{x_1,\ldots,x_m\}$ we use the empirical pseudo-metrics
$$\mathrm{dist}_{L_1(S)}(g,g') := \frac1m\sum_{i=1}^m|g(x_i)-g'(x_i)|, \qquad \mathrm{dist}_{L_2(S)}(g,g') := \Big(\frac1m\sum_{i=1}^m\big(g(x_i)-g'(x_i)\big)^2\Big)^{1/2},$$
and write $\|g\|_{S} := \mathrm{dist}_{L_2(S)}(g,0)$. Covering numbers are those of Definition 7.4.1, with cover elements taken from the class itself.

**Lemma 7.6.1 (Packing bound for VC classes).** Let $\mathcal{G}$ be a class of $\{0,1\}$-valued functions with VC dimension $d \geq 1$, and let $P$ be any distribution on the domain, with $\mathrm{dist}_{L_1(P)}(g,g') := \mathbb{E}_{x\sim P}|g(x)-g'(x)| = P\big(g(x) \neq g'(x)\big)$. Then for every $\epsilon \in (0,1]$,
$$\log N\big(\epsilon,\mathcal{G},L_1(P)\big) \;\leq\; 2d\log\frac{4e}{\epsilon}.$$
In particular this holds for $L_1(S)$, taking $P$ uniform on the points of $S$, uniformly over all samples $S$.

**Proof.** Call $g_1,\ldots,g_M \in \mathcal{G}$ **$\epsilon$-separated** if $\mathrm{dist}_{L_1(P)}(g_j,g_k) > \epsilon$ for all $j \neq k$. We show that every $\epsilon$-separated family satisfies $\log M \leq 2d\log(4e/\epsilon)$. Granting this, a separated family of maximal size exists, and as in the proof of Lemma 7.4.3 it is an $\epsilon$-cover (a function farther than $\epsilon$ from all its members could be added to it), which proves the lemma.

Let $M \geq 2$ (for $M = 1$ there is nothing to prove), set $n := \lceil 2\log M/\epsilon \rceil$, and draw $x_1,\ldots,x_n$ i.i.d. from $P$. For a fixed pair $j < k$, the two functions agree on all $n$ points with probability
$$\big(1 - \mathrm{dist}_{L_1(P)}(g_j,g_k)\big)^n < (1-\epsilon)^n \leq e^{-\epsilon n} \leq M^{-2}.$$
There are fewer than $M^2/2$ pairs, so by the union bound the probability that *some* pair agrees on all $n$ points is less than $\tfrac12$. Hence some realization of the $n$ points — a set of at most $n$ distinct points — separates all $M$ functions, so that $M$ is at most the number of distinct restrictions of $\mathcal{G}$ to that set, i.e. $M \leq \tau_{\mathcal{G}}(n)$ (Definition 6.4.4; $\tau$ is non-decreasing, so having fewer than $n$ distinct points only helps).

If $n \leq d$, then $M \leq 2^n \leq 2^d$ and $\log M \leq d\log 2$, which is below the claim. If $n > d$, Lemma 6.4.8 gives $M \leq (en/d)^d$. Write $u := \log M/d$. Since $n \leq 2\log M/\epsilon + 1 = 2du/\epsilon + 1$ and $\epsilon \leq 1 \leq d$,
$$u \;\leq\; \log\frac{e\,n}{d} \;\leq\; \log\Big(e\Big(\frac{2u}{\epsilon}+\frac1d\Big)\Big) \;\leq\; 1 + \log\frac1\epsilon + \log(2u+1).$$
The function $u \mapsto \tfrac u2 + \log 4 - \log(2u+1)$ has derivative $\tfrac12 - \tfrac{2}{2u+1}$, hence its minimum over $u \geq 0$ at $u = \tfrac32$, where its value is $\tfrac34 > 0$; so $\log(2u+1) \leq \tfrac u2 + \log 4$. Substituting, $u \leq 1 + \log\frac1\epsilon + \log 4 + \frac u2$, i.e. $u \leq 2\log\frac{4e}{\epsilon}$ $\square$

The proof is the probabilistic method in its plainest form: a random sample of $O(\log M/\epsilon)$ points tells $M$ well-separated functions apart, so $M$ cannot exceed the number of labelings a VC class produces on that many points. The volumetric bound of Lemma 7.4.3 needed the dimension of a parameter space; Lemma 7.6.1 needs only the VC dimension, and it holds for every sample at once. A more delicate argument due to [Haussler (1995)](https://doi.org/10.1016/0097-3165(95)90052-7)<!-- cite: haussler1995sphere | article | author={Haussler, David}; title={Sphere Packing Numbers for Subsets of the Boolean Cube with Bounded Vapnik-Chervonenkis Dimension}; journal={Journal of Combinatorial Theory, Series A}; year={1995}; volume={69}; number={2}; pages={217--232} --> improves the right-hand side to $1 + \log(d+1) + d\log(2e/\epsilon)$; for our purposes the difference is a constant factor.

For $\{0,1\}$-valued functions $(g(x)-g'(x))^2 = |g(x)-g'(x)|$, so $\mathrm{dist}_{L_2(S)} = \sqrt{\mathrm{dist}_{L_1(S)}}$ and an $L_1(S)$-cover at scale $\epsilon^2$ is an $L_2(S)$-cover at scale $\epsilon$:
$$\log N\big(\epsilon,\mathcal{G},L_2(S)\big) \;\leq\; 2d\log\frac{4e}{\epsilon^2} \qquad (\epsilon \in (0,1]). \tag{7.2}$$

**Theorem 7.6.2 (Dudley's entropy integral).** Let $\mathcal{G}$ be a class of real-valued functions and $S$ a sample of size $m$ with $B := \sup_{g\in\mathcal{G}}\|g\|_S < \infty$. Then
$$\widehat{\mathfrak{R}}_S(\mathcal{G}) \;\leq\; \inf_{\epsilon_0 \geq 0}\ \Big[\,4\epsilon_0 + \frac{12}{\sqrt m}\int_{\epsilon_0}^{\infty}\sqrt{\log N\big(\epsilon,\mathcal{G},L_2(S)\big)}\,d\epsilon\Big].$$

**Proof.** Identify each $g$ with its vector $(g(x_1),\ldots,g(x_m)) \in \mathbb{R}^m$, and note that for any vector $v$ and any signs, Cauchy-Schwarz gives
$$\frac1m\langle\sigma,v\rangle \;\leq\; \frac1m\|\sigma\|_2\|v\|_2 = \frac1m\sqrt m\cdot\sqrt m\,\|v\|_S = \|v\|_S. \tag{7.3}$$
*Step 1: a chain of covers.* Let $\epsilon_\ell := 2^{-\ell}B$ for $\ell = 0,1,2,\ldots$, let $\mathcal{G}_0 := \{0\}$, and for $\ell \geq 1$ let $\mathcal{G}_\ell \subseteq \mathcal{G}$ be an $\epsilon_\ell$-cover of minimal size $N_\ell := N(\epsilon_\ell,\mathcal{G},L_2(S))$. For each $g$ pick $g_\ell \in \mathcal{G}_\ell$ with $\|g - g_\ell\|_S \leq \epsilon_\ell$; for $\ell = 0$ this holds with $g_0 = 0$ by definition of $B$. For any $L \geq 1$, telescoping gives
$$g = (g - g_L) + \sum_{\ell=1}^L (g_\ell - g_{\ell-1}).$$
*Step 2: split the supremum.* A supremum of a sum is at most the sum of the suprema, so
$$\widehat{\mathfrak{R}}_S(\mathcal{G}) \;\leq\; \mathbb{E}_\sigma\Big[\sup_g\frac1m\langle\sigma, g - g_L\rangle\Big] + \sum_{\ell=1}^L \mathbb{E}_\sigma\Big[\sup_g\frac1m\langle\sigma, g_\ell - g_{\ell-1}\rangle\Big].$$
By (7.3) the first term is at most $\epsilon_L$. In the $\ell$-th term the vector $g_\ell - g_{\ell-1}$ takes at most $N_\ell N_{\ell-1} \leq N_\ell^2$ distinct values (covering numbers do not increase with the scale, so $N_{\ell-1} \leq N_\ell$), each of norm $\|g_\ell - g_{\ell-1}\|_S \leq \|g_\ell - g\|_S + \|g - g_{\ell-1}\|_S \leq \epsilon_\ell + \epsilon_{\ell-1} = 3\epsilon_\ell$, i.e. of Euclidean norm at most $3\epsilon_\ell\sqrt m$. Massart's lemma (Lemma 7.3.3) bounds it by
$$\frac{3\epsilon_\ell\sqrt m\,\sqrt{2\log N_\ell^2}}{m} \;=\; 6\,\epsilon_\ell\sqrt{\frac{\log N_\ell}{m}} \;=\; \frac{12}{\sqrt m}(\epsilon_\ell - \epsilon_{\ell+1})\sqrt{\log N(\epsilon_\ell)} \;\leq\; \frac{12}{\sqrt m}\int_{\epsilon_{\ell+1}}^{\epsilon_\ell}\sqrt{\log N(\epsilon)}\,d\epsilon,$$
using $\epsilon_\ell = 2(\epsilon_\ell - \epsilon_{\ell+1})$ and, once more, that $N(\epsilon) \geq N(\epsilon_\ell)$ for $\epsilon \leq \epsilon_\ell$. Summing over $\ell = 1,\ldots,L$,
$$\widehat{\mathfrak{R}}_S(\mathcal{G}) \;\leq\; \epsilon_L + \frac{12}{\sqrt m}\int_{\epsilon_{L+1}}^{\infty}\sqrt{\log N(\epsilon)}\,d\epsilon.$$
*Step 3: choose $L$.* If $\epsilon_0 = 0$, let $L \to \infty$. If $\epsilon_0 \geq B/4$, the claim holds trivially, since (7.3) gives $\widehat{\mathfrak{R}}_S(\mathcal{G}) \leq B \leq 4\epsilon_0$. Otherwise $\epsilon_2 = B/4 > \epsilon_0$, and we take $L+1$ to be the largest index with $\epsilon_{L+1} > \epsilon_0$, so that $L \geq 1$, $\epsilon_{L+1} > \epsilon_0$ and $\epsilon_{L+2} \leq \epsilon_0$; then $\epsilon_L = 4\epsilon_{L+2} \leq 4\epsilon_0$ and the integral over $[\epsilon_{L+1},\infty)$ is at most the integral over $[\epsilon_0,\infty)$ $\square$

The proof is Massart's lemma applied once per scale. A single-scale argument pays $\sqrt{\log N(\epsilon)}$ times the full radius $B$ of the class; chaining pays at each scale only for the *increments* $g_\ell - g_{\ell-1}$, whose radius $3\epsilon_\ell$ shrinks geometrically while their number grows. The resulting series is a Riemann sum of the entropy integral. The bound goes back to [Dudley (1967)](https://doi.org/10.1016/0022-1236(67)90017-1)<!-- cite: dudley1967sizes | article | author={Dudley, Richard M.}; title={The Sizes of Compact Subsets of Hilbert Space and Continuity of Gaussian Processes}; journal={Journal of Functional Analysis}; year={1967}; volume={1}; number={3}; pages={290--330} -->; Section 27.2 of [Shalev-Shwartz & Ben-David (2014)](https://doi.org/10.1017/CBO9781107298019)<!-- cite: shalev2014understanding | book | author={Shalev-Shwartz, Shai and Ben-David, Shai}; title={Understanding Machine Learning: From Theory to Algorithms}; publisher={Cambridge University Press}; year={2014} --> gives a closely related form, and the one above is Theorem 6.25 of [Zhang (2023)](https://doi.org/10.1017/9781009093057)<!-- cite: zhang2023mathematical | book | author={Zhang, Tong}; title={Mathematical Analysis of Machine Learning Algorithms}; publisher={Cambridge University Press}; year={2023} -->.

---

**Corollary 7.6.3 (Sharp Rademacher bound for VC classes).** Let $\mathcal{G}$ be a class of $\{0,1\}$-valued functions with VC dimension $d \geq 1$. Then for **every** sample $S$ of size $m$,
$$\widehat{\mathfrak{R}}_S(\mathcal{G}) \;\leq\; 48\sqrt{\frac{d}{m}}.$$

**Proof.** Apply Theorem 7.6.2 with $\epsilon_0 = 0$. Two $\{0,1\}$-valued functions are within $L_2(S)$-distance $1$ of each other, so $N(\epsilon,\mathcal{G},L_2(S)) = 1$ for $\epsilon \geq 1$ and the integral runs over $(0,1]$ only. There, by (7.2) and $\sqrt{u+v} \leq \sqrt u + \sqrt v$,
$$\sqrt{\log N\big(\epsilon,\mathcal{G},L_2(S)\big)} \;\leq\; \sqrt{2d}\,\Big(\sqrt{\log 4e} + \sqrt{2}\sqrt{\log\tfrac1\epsilon}\Big).$$
The substitution $\epsilon = e^{-s}$, $d\epsilon = -e^{-s}ds$ gives
$$\int_0^1\sqrt{\log\tfrac1\epsilon}\,d\epsilon = \int_0^\infty \sqrt{s}\,e^{-s}\,ds = \Gamma\big(\tfrac32\big) = \frac{\sqrt\pi}{2},$$
so
$$\widehat{\mathfrak{R}}_S(\mathcal{G}) \;\leq\; \frac{12\sqrt{2d}}{\sqrt m}\Big(\sqrt{\log 4e} + \sqrt2\,\frac{\sqrt\pi}{2}\Big) \;\leq\; 12\sqrt2\,(1.545 + 1.254)\sqrt{\frac dm} \;\leq\; 47.5\sqrt{\frac dm}. \qquad\square$$

The constant is not optimized; Haussler's sharper covering estimate, for instance, reduces it.

---

This removes the $\sqrt{\log(m+1)}$ factor of Theorem 7.3.5: covering a VC class at *every* resolution simultaneously costs only a constant factor more than the single-resolution argument. The zero-one loss $\ell(y,h(x)) = |h(x) - y|$ is $1$-Lipschitz in $h(x)$, so the contraction lemma (Lemma 7.3.4) gives $\widehat{\mathfrak{R}}_S(\ell\circ\mathcal{H}) \leq \widehat{\mathfrak{R}}_S(\mathcal{H})$, and feeding Corollary 7.6.3 into Theorem 7.3.2 gives, for a class $\mathcal{H}$ of binary classifiers with $d_{VC}(\mathcal{H}) = d$, with probability at least $1-\delta$, simultaneously for all $h$,
$$R(h) \;\leq\; \widehat{R}_S(h) + 96\sqrt{\frac{d}{m}} + 3\sqrt{\frac{\log(2/\delta)}{2m}}.$$
The same argument applied to the loss $1 - \ell$, whose Rademacher complexity equals that of $\ell$ (replace $\sigma$ by $-\sigma$), bounds $\widehat{R}_S(h) - R(h)$ identically; with $\delta/2$ for each side,
$$\sup_{h\in\mathcal{H}}\big|R(h) - \widehat{R}_S(h)\big| \;\leq\; 96\sqrt{\frac{d}{m}} + 3\sqrt{\frac{\log(4/\delta)}{2m}} \qquad \text{w.p.} \geq 1-\delta, \tag{7.4}$$
which is the uniform-convergence upper bound of the Fundamental Theorem (Theorem 6.4.7) without the residual $\log(1/\epsilon)$ factor.

# Sparse Combinations: Maurey's Lemma

The convex hull will appear in Chapter 12's analysis of boosting as the closure of the class the ensemble actually searches. Its Rademacher complexity is that of the base class (Theorem 12.2.6), but a boosting ensemble run for $T$ rounds uses only $T$ members — a *sparse* point of the hull, not an arbitrary one. Two results connect the sparse and dense viewpoints, and both will be consumed by Chapters 10 and 12.

The first result is a lemma of Maurey, published by [Pisier (1981)](http://www.numdam.org/item/SAF_1980-1981____A5_0/)<!-- cite: pisier1981remarques | inproceedings | author={Pisier, Gilles}; title={Remarques sur un r{\'e}sultat non publi{\'e} de {B. Maurey}}; booktitle={S{\'e}minaire d'Analyse Fonctionnelle (Polytechnique)}; year={1981}; note={Expos{\'e} no. 5} --> and brought to approximation and learning theory by [Jones (1992)](https://doi.org/10.1214/aos/1176348546)<!-- cite: jones1992simple | article | author={Jones, Lee K.}; title={A Simple Lemma on Greedy Approximation in Hilbert Space and Convergence Rates for Projection Pursuit Regression and Neural Network Training}; journal={Annals of Statistics}; year={1992}; volume={20}; number={1}; pages={608--613} --> and [Barron (1993)](https://doi.org/10.1109/18.256500)<!-- cite: barron1993universal | article | author={Barron, Andrew R.}; title={Universal Approximation Bounds for Superpositions of a Sigmoidal Function}; journal={IEEE Transactions on Information Theory}; year={1993}; volume={39}; number={3}; pages={930--945} -->.

**Theorem 7.7.1 (Maurey's sparse approximation).** Fix a sample $x_1,\ldots,x_m$ and let $\Psi$ be a family of functions with $\big(\frac1m\sum_{i=1}^m\psi(x_i)^2\big)^{1/2} \leq B$ for every $\psi \in \Psi$. Let $f = \sum_{j=1}^n w_j\,\psi_j$ be a finite combination with $\psi_j \in \Psi$ and total weight $A := \sum_j |w_j|$. Then for every integer $N \geq 1$ there exist indices $j_1,\ldots,j_N$ such that $f_N := \frac{A}{N}\sum_{k=1}^N u_{j_k}\psi_{j_k}$ — where $u_j := \mathrm{sign}(w_j)$, so that $u_{j_k}\psi_{j_k}$ is a member of the *symmetrized* class $\Psi \cup (-\Psi)$ — satisfies
$$\frac1m\sum_{i=1}^m\big(f_N(x_i) - f(x_i)\big)^2 \;\leq\; \frac{A^2 B^2}{N}.$$

**Proof.** Assume $A > 0$ (otherwise $f = 0$ and $f_N = 0$ works) and let $J_1,\ldots,J_N$ be i.i.d. indices with $P(J_k = j) = |w_j|/A$. Define the random function $f_N^{(J)} := \frac{A}{N}\sum_{k=1}^N u_{J_k}\psi_{J_k}$. For every fixed $x$,
$$\mathbb{E}_J\big[f_N^{(J)}(x)\big] = \frac{A}{N}\sum_{k=1}^N\mathbb{E}\big[u_{J_k}\psi_{J_k}(x)\big] = A\sum_j \frac{|w_j|}{A}\,\mathrm{sign}(w_j)\psi_j(x) = \sum_j w_j\psi_j(x) = f(x),$$
so $f_N^{(J)}$ is an unbiased estimator of $f$ at every point, and an average of $N$ i.i.d. terms $A\,u_{J_k}\psi_{J_k}(x)$. Its variance at each $x_i$ is therefore $\frac{1}{N}\mathrm{Var}\big[A\,u_{J_1}\psi_{J_1}(x_i)\big] \leq \frac{A^2}{N}\mathbb{E}\big[\psi_{J_1}(x_i)^2\big]$, and averaging over the sample,
$$\mathbb{E}_J\Big[\frac1m\sum_i\big(f_N^{(J)}(x_i) - f(x_i)\big)^2\Big] \;=\; \frac{1}{m}\sum_i \mathrm{Var}_J\big[f_N^{(J)}(x_i)\big] \;\leq\; \frac{A^2}{N}\,\mathbb{E}_J\Big[\frac1m\sum_i \psi_{J_1}(x_i)^2\Big] \;\leq\; \frac{A^2B^2}{N}.$$
A random variable cannot exceed its mean with probability one, so some realization $J$ satisfies the claimed bound $\square$

In words: **any point of the convex hull ($A = 1$) is, on any finite sample, within $B/\sqrt N$ of some $N$-term average of base functions.** Sparsity costs only $1/\sqrt N$ — the same rate at which an i.i.d. average concentrates around its mean, which is exactly what the proof exploits. For a loss $L(f,y)$ that is convex and $\beta$-smooth in $f$ (its derivative in $f$ is $\beta$-Lipschitz), the excess empirical loss of $f_N$ over $f$ is, on average over $J$, at most $\frac{\beta}{2}$ times the squared distance above: the first-order term of the Taylor expansion averages to zero because $f_N^{(J)}$ is unbiased. So some $N$-sparse combination is within $\beta A^2B^2/(2N)$ of any point of the hull in empirical loss. Gradient boosting (Theorem 12.3.1 in Chapter 12) finds such a sparse combination greedily, at the same $O(1/N)$ rate.

The complementary statement is that *constraining the combination weights to an $L_1$ ball does not increase the Rademacher complexity at all*:

**Theorem 7.7.2 (Rademacher complexity of $L_1$-constrained combinations).** Let $\Psi$ be a symmetric class of functions ($\psi \in \Psi \Rightarrow -\psi \in \Psi$) and let
$$\mathcal{F}_{A,L_1}(\Psi) := \Big\{x \mapsto \sum_{j=1}^k w_j\,\psi_j(x) \;:\; k \in \mathbb{N},\ \psi_j \in \Psi,\ w_j \in \mathbb{R},\ \sum_j |w_j| \leq A\Big\}.$$
Then for every sample $S$,
$$\widehat{\mathfrak{R}}_S\big(\mathcal{F}_{A,L_1}(\Psi)\big) \;=\; A\cdot\widehat{\mathfrak{R}}_S(\Psi).$$

**Proof.** "$\geq$": every singleton combination $A\cdot\psi$, $\psi \in \Psi$, lies in $\mathcal{F}_{A,L_1}(\Psi)$, so for every realization of the signs the supremum over $\mathcal{F}_{A,L_1}(\Psi)$ is at least $\sup_{\psi\in\Psi}\frac{A}{m}\sum_i\sigma_i\psi(x_i)$; taking the expectation over $\sigma$ gives $\widehat{\mathfrak{R}}_S(\mathcal{F}_{A,L_1}(\Psi)) \geq A\,\widehat{\mathfrak{R}}_S(\Psi)$.
"$\leq$": fix a realization of the signs $\sigma$ and a combination $g = \sum_j w_j\psi_j$ with $\sum_j|w_j| \leq A$. Then
$$\Big|\frac1m\sum_i \sigma_i g(x_i)\Big| = \Big|\sum_j w_j\,\frac1m\sum_i\sigma_i\psi_j(x_i)\Big| \leq \sum_j |w_j|\,\Big|\frac1m\sum_i\sigma_i\psi_j(x_i)\Big| \leq \Big(\sum_j |w_j|\Big)\sup_{\psi\in\Psi}\Big|\frac1m\sum_i\sigma_i\psi(x_i)\Big|,$$
and since $\Psi$ is symmetric, $\sup_{\psi\in\Psi}\big|\frac1m\sum_i\sigma_i\psi(x_i)\big| = \sup_{\psi\in\Psi}\frac1m\sum_i\sigma_i\psi(x_i)$: the absolute value of a negative correlation is attained, with positive sign, by $-\psi \in \Psi$. Taking the supremum over $g$ and then the expectation over $\sigma$ gives $\widehat{\mathfrak{R}}_S(\mathcal{F}_{A,L_1}(\Psi)) \leq A\,\widehat{\mathfrak{R}}_S(\Psi)$ $\square$

Note that the number of terms $k$ and the individual weights never appear in the bound — only the total $L_1$ mass $A$ does. Since $\mathrm{conv}(\Psi) \subseteq \mathcal{F}_{1,L_1}(\Psi)$, the upper bound in Theorem 12.2.6's "convexification is free" is, for a symmetric base class, the $A=1$ instance of this statement, and the two proofs are the same argument. What the $L_1$ form adds is the **norm constraint** reading: capacity is controlled by the *budget* $A$, not by the number of parameters $k$ — the same moral as Theorem 7.3.6 for $L_2$-balls, and the same mechanism we will exploit for the output layer of a two-layer network in Chapter 10 and for gradient boosting in Chapter 12.

# Algorithmic Stability

Every measure so far prices the *class* $\mathcal{H}$: the same complexity is charged no matter which algorithm searched it, and no credit is given for the algorithm's actual behavior on the data. **Algorithmic stability** is the complementary theory: it prices the *algorithm*, by asking how much its output can change when a single training example is swapped. We develop it in the distribution-free, expected-risk form initiated by [Bousquet & Elisseeff (2002)](https://www.jmlr.org/papers/v2/bousquet02a.html)<!-- cite: bousquet2002stability | article | author={Bousquet, Olivier and Elisseeff, St{\'e}phane}; title={Stability and Generalization}; journal={Journal of Machine Learning Research}; year={2002}; volume={2}; pages={499--526} -->.

Throughout this section, a learning algorithm $A$ maps a training set $S$ to a hypothesis $A(S)$, possibly using internal randomness (as stochastic gradient descent does); $\mathbb{E}_A$ denotes the average over that randomness, and is void for a deterministic $A$; and $\phi(\cdot,z)$ denotes the loss of a hypothesis on example $z$ — the test loss may differ from the training objective, as in regularized ERM.

**Definition 7.8.1 (Uniform stability).** $A$ is $\epsilon$-uniformly stable if for all training sets $S, S'$ that differ in exactly one element,
$$\sup_{z}\ \big|\mathbb{E}_A\,\phi(A(S),z) - \mathbb{E}_A\,\phi(A(S'),z)\big| \;\leq\; \epsilon.$$

---

In words: swapping any one training example cannot move any *single* prediction's loss by more than $\epsilon$. Stability is a property of the algorithm's input-output map, not of the size of the class it searches; an algorithm that memorizes $S$ is maximally unstable (swapping a point changes its prediction on that point outright), while an algorithm that averages over data has its outputs anchored by the $m-1$ untouched examples.

**Theorem 7.8.2 (Stability implies generalization).** If $A$ is $\epsilon$-uniformly stable, then
$$\mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_A\big[R(A(S))\big] \;\leq\; \mathbb{E}_{S\sim\mathcal{D}^m}\mathbb{E}_A\big[\widehat{R}_S(A(S))\big] + \epsilon.$$

**Proof.** We write the proof for deterministic $A$; for a randomized one, replace $\phi(A(S),z)$ by $\mathbb{E}_A\,\phi(A(S),z)$ throughout, which is exactly the quantity Definition 7.8.1 controls. Write $S = (z_1,\ldots,z_m)$ and let $z$ be a fresh example. Since $R(A(S)) = \mathbb{E}_{z}\big[\phi(A(S),z)\big]$, it suffices to compare the two expectations term by term. For each $i \in [m]$, consider the **swap map** $T_i$ on $(m{+}1)$-tuples of examples that exchanges $z_i$ and $z$,
$$(z_1,\ldots,z_{i-1},\ z_i,\ z_{i+1},\ldots,z_m,\ z)\ \mapsto\ (z_1,\ldots,z_{i-1},\ z,\ z_{i+1},\ldots,z_m,\ z_i).$$
All $m+1$ coordinates being i.i.d. from $\mathcal{D}$, the law of the tuple is invariant under $T_i$, so for any function $G$ of the tuple, $\mathbb{E}[G] = \mathbb{E}[G\circ T_i]$. Applying this to $G := \phi(A(S),z)$ (which depends on the first $m$ coordinates only, through $A$) gives
$$\mathbb{E}_{S,z}\big[\phi(A(S),z)\big] \;=\; \mathbb{E}_{S,z}\big[\phi\big(A(S^{(i)}),\,z_i\big)\big],$$
where $S^{(i)}$ is $S$ with its $i$-th element replaced by $z$. Now $S^{(i)}$ and $S$ differ in exactly one element, so by uniform stability $\phi(A(S^{(i)}),z_i) \leq \phi(A(S),z_i) + \epsilon$ pointwise. Averaging over $i$,
$$\mathbb{E}_S\big[R(A(S))\big] = \frac1m\sum_{i=1}^m \mathbb{E}_{S,z}\big[\phi\big(A(S^{(i)}),z_i\big)\big] \;\leq\; \frac1m\sum_{i=1}^m\mathbb{E}_{S,z}\big[\phi(A(S),z_i) + \epsilon\big] = \mathbb{E}_S\big[\widehat{R}_S(A(S))\big] + \epsilon. \qquad\square$$

The proof is a pure **ghost-sample swap**: the fresh example $z$ and the training example $z_i$ are statistically exchangeable, so the algorithm's loss on the fresh example can be evaluated as its loss on the replaced example — where stability bridges the two runs. No complexity measure of any class appears anywhere; the entire price of generalization is the one-sample sensitivity $\epsilon$.

A function $\mathcal{J}$ on a convex set $\Omega$ is **$\lambda$-strongly convex** if for all $w,u \in \Omega$ and $\alpha \in (0,1)$,
$$\mathcal{J}\big(\alpha w + (1-\alpha)u\big) \;\leq\; \alpha\,\mathcal{J}(w) + (1-\alpha)\,\mathcal{J}(u) - \frac\lambda2\,\alpha(1-\alpha)\|w-u\|_2^2;$$
for $\lambda = 0$ this is convexity. The regularizer $\frac\lambda2\|w\|_2^2$ is $\lambda$-strongly convex (the inequality holds with equality, as expanding the squares shows), and adding a convex function preserves $\lambda$-strong convexity. If $w^*$ minimizes a $\lambda$-strongly convex $\mathcal{J}$ over $\Omega$, then for every $w \in \Omega$, using $\mathcal{J}(w^*) \leq \mathcal{J}(\alpha w + (1-\alpha)w^*)$ in the definition, dividing by $\alpha$ and letting $\alpha \to 0$,
$$\mathcal{J}(w) - \mathcal{J}(w^*) \;\geq\; \frac\lambda2\|w - w^*\|_2^2. \tag{7.5}$$
In particular a minimizer, when it exists, is unique.

**Theorem 7.8.3 (Regularized ERM is uniformly stable).** Let $\Omega$ be a closed convex set, $\phi(w,z)$ a convex function of $w \in \Omega$ that is $G$-Lipschitz in $w$ for every $z$, and $h \geq 0$ a $\lambda$-strongly convex regularizer, such that the regularized ERM
$$A(S) \;=\; \arg\min_{w\in\Omega}\ \mathcal{J}_S(w) := \frac1m\sum_{i=1}^m \phi(w,z_i) + h(w)$$
exists for every $S$. Then $A$ is $\frac{2G^2}{\lambda m}$-uniformly stable.

**Proof.** Let $S'$ differ from $S$ in the $i$-th element only ($z_i'$ instead of $z_i$), and let $w := A(S)$, $w' := A(S')$. Both $\mathcal{J}_S$ and $\mathcal{J}_{S'}$ are $\lambda$-strongly convex, so (7.5) applied to each at its own minimizer gives
$$\mathcal{J}_S(w') - \mathcal{J}_S(w) \;\geq\; \frac\lambda2\|w'-w\|_2^2, \qquad \mathcal{J}_{S'}(w) - \mathcal{J}_{S'}(w') \;\geq\; \frac\lambda2\|w'-w\|_2^2.$$
The two objectives differ only in the $i$-th term, $\mathcal{J}_S - \mathcal{J}_{S'} = \frac1m\big(\phi(\cdot,z_i) - \phi(\cdot,z_i')\big)$, so adding the two inequalities cancels all remaining terms:
$$\lambda\|w'-w\|_2^2 \;\leq\; \frac{\phi(w',z_i) - \phi(w,z_i)}{m} + \frac{\phi(w,z_i') - \phi(w',z_i')}{m} \;\leq\; \frac{2G}{m}\|w'-w\|_2,$$
by the Lipschitz property. Hence $\|w - w'\|_2 \leq \frac{2G}{\lambda m}$, and for every test example $z$, again by the Lipschitz property, $\big|\phi(w,z) - \phi(w',z)\big| \leq G\,\|w - w'\|_2 \leq \frac{2G^2}{\lambda m}$ $\square$

The structural content is that **strong convexity converts a perturbation of the objective that is $\frac{2G}{m}$-Lipschitz into a movement of the minimizer of at most $\frac{2G}{\lambda m}$** — the regularization strength $\lambda$ is exactly what provides stability, at the price of the approximation error $h(w)$. The argument is that of Section 13.3 of [Shalev-Shwartz & Ben-David (2014)](https://doi.org/10.1017/CBO9781107298019)<!-- cite: shalev2014understanding | book | author={Shalev-Shwartz, Shai and Ben-David, Shai}; title={Understanding Machine Learning: From Theory to Algorithms}; publisher={Cambridge University Press}; year={2014} -->, following Bousquet & Elisseeff. Nothing in it uses finite dimension: it holds verbatim in any Hilbert space, in particular in an RKHS (Chapter 11).

Combining the two theorems gives the standard **expected oracle inequality** for regularized ERM:
$$\mathbb{E}_S\big[R(A(S))\big] \;\leq\; \mathbb{E}_S\big[\mathcal{J}_S(A(S))\big] + \frac{2G^2}{\lambda m} \;\leq\; \inf_{w\in\Omega}\Big[\mathbb{E}_z\big[\phi(w,z)\big] + h(w)\Big] + \frac{2G^2}{\lambda m},$$
where $R(w) := \mathbb{E}_z[\phi(w,z)]$ and $\widehat{R}_S(w) := \frac1m\sum_i\phi(w,z_i)$, the first inequality is Theorem 7.8.2 together with $h\geq 0$ (the training objective upper-bounds the training risk), and the second is the optimality of $A(S)$ for $\mathcal{J}_S$ plus $\mathbb{E}_S[\mathcal{J}_S(w)] = \mathbb{E}_z[\phi(w,z)] + h(w)$. Read at the optimal $w$, the excess expected risk of regularized ERM over the best regularized population objective is $O(G^2/(\lambda m))$: an $O(1/m)$ rate, in contrast to the $O(1/\sqrt m)$ of the complexity-based bounds — the price being that the bound is in expectation over $S$, and is relative to the regularized objective rather than the best-in-class risk.

Two caveats. First, the expectation over $S$ in Theorem 7.8.2 can be upgraded to a high-probability statement. Replacing one example changes $R(A(S)) - \widehat{R}_S(A(S))$ by at most $2\epsilon + M/m$ for a loss with range $M$, so McDiarmid's inequality applies; this is the route of Bousquet & Elisseeff, and the resulting bound is informative only when $\epsilon = o(1/\sqrt m)$. Bounds with a nearly optimal dependence on $\epsilon$ require a more refined concentration argument, see [Feldman & Vondrák (2019)](https://proceedings.mlr.press/v99/feldman19a.html)<!-- cite: feldman2019high | inproceedings | author={Feldman, Vitaly and Vondr{\'a}k, Jan}; title={High Probability Generalization Bounds for Uniformly Stable Algorithms with Nearly Optimal Rate}; booktitle={Proceedings of the 32nd Conference on Learning Theory}; series={PMLR}; volume={99}; year={2019}; pages={1270--1279} -->. Second, the oracle inequality bounds the excess risk against $\inf_w [R(w) + h(w)]$, in which the regularization bias is paid on the inside; comparing against $\inf_{w\in\Omega} R(w)$ instead requires balancing $\lambda$ against $m$ and re-optimizing — the same bias-complexity trade-off, now with $\lambda$ in the role that $\log|\mathcal{H}|$ played in Chapter 6.

# Minimax Lower Bounds

Every bound in this chapter so far is an *upper* bound: it certifies that some algorithm achieves at most a certain rate. The complementary question is whether that rate is *necessary*. The right framework is minimax: fix the sample size $m$, let the algorithm be chosen *before* the distribution, and ask what the best algorithm can guarantee against the *worst* distribution.

**Definition 7.9.1 (Minimax risk).** For a distribution family $\mathcal{P}$ over examples, a quality measure $Q$ assigning to each pair (hypothesis, distribution) a nonnegative loss, and an algorithm $A$ mapping training sets to hypotheses, the worst-case risk of $A$ is
$$r_m(A,\mathcal{P},Q) := \sup_{\mathcal{D}\in\mathcal{P}}\ \mathbb{E}_{S\sim\mathcal{D}^m}\ \mathbb{E}_A\big[Q(A(S),\mathcal{D})\big],$$
where $\mathbb{E}_A$ averages over any internal randomization of $A$. The **minimax risk** is $r_m(\mathcal{P},Q) := \inf_A r_m(A,\mathcal{P},Q)$, the infimum over all algorithms. A lower bound $r_m(\mathcal{P},Q) \geq c$ certifies that *no* algorithm — however clever — achieves quality better than $c$ on every $\mathcal{D}\in\mathcal{P}$; matching it with an upper bound identifies $r_m$ up to constants and certifies that the upper-bound proof technique was not merely loose.

The elementary engine of such lower bounds is the **two-point method** (Le Cam's method): to show that $m$ samples do not suffice, exhibit two distributions that are (i) statistically almost indistinguishable from $m$ samples, yet (ii) demand conflicting predictions. Its measure of indistinguishability is the **total variation distance**
$$\mathrm{TV}(P,Q) := \sup_{E}\big|P(E) - Q(E)\big|,$$
whose single property we need is the change-of-measure bound
$$\big|\mathbb{E}_P[g] - \mathbb{E}_Q[g]\big| \;\leq\; \mathrm{TV}(P,Q) \qquad \text{for every } g \text{ with } 0 \leq g \leq 1, \tag{7.6}$$
which follows from the likelihood-ratio representation of TV. When $P \ll Q$ with likelihood ratio $L := \frac{dP}{dQ}$, every expectation difference is an expectation under $Q$:
$$\mathbb{E}_P[g] - \mathbb{E}_Q[g] = \mathbb{E}_Q\big[(L-1)\,g\big] \;\leq\; \mathbb{E}_Q\big[(L-1)^+\big],$$
since $g \leq 1$ makes $(L-1)\,g \leq (L-1)^+$ pointwise (on $\{L \leq 1\}$ the left side is $\leq 0$). The quantity $\mathbb{E}_Q[(L-1)^+]$ equals $\mathrm{TV}(P,Q)$: the supremum in the definition is attained by the event $E^* = \{L > 1\}$, where $P(E^*) - Q(E^*) = \mathbb{E}_Q[(L-1)\mathds{1}_{\{L>1\}}] = \mathbb{E}_Q[(L-1)^+]$, and $\mathbb{E}_Q[L-1] = 0$ makes the negative part of $L-1$ equal in magnitude to the positive part, so no event does better in absolute value either.

**Lemma 7.9.2 (Two-point testing bounds).** Let $P,Q$ be two distributions on a common space and $W$ any event.

*(i)* $P(W) + Q(W^c) \;\geq\; 1 - \mathrm{TV}(P,Q)$.

*(ii)* For product distributions of $m$ i.i.d. draws, with $\chi^2(P\|Q) := \mathbb{E}_Q\big[\big(\tfrac{dP}{dQ}-1\big)^2\big]$ the **chi-squared divergence**,
$$\mathrm{TV}\big(P^{\otimes m},Q^{\otimes m}\big) \;\leq\; \sqrt{\chi^2\big(P^{\otimes m}\|Q^{\otimes m}\big)} \;\leq\; \sqrt{2m\,\chi^2(P\|Q)} \qquad \text{whenever } m\,\chi^2(P\|Q) \leq 1.$$

*(iii)* (Bretagnolle-Huber) With $\mathrm{KL}(P\|Q) := \mathbb{E}_P\big[\log\frac{dP}{dQ}\big]$ the Kullback-Leibler divergence of the PAC-Bayes section,
$$P(W) + Q(W^c) \;\geq\; \tfrac12\,e^{-\mathrm{KL}(P\|Q)}, \qquad \mathrm{KL}\big(P^{\otimes m}\big\|Q^{\otimes m}\big) = m\,\mathrm{KL}(P\|Q).$$

**Proof.** *(i)* By the change-of-measure bound (7.6) with $g := \mathds{1}_W$, $P(W) \geq Q(W) - \mathrm{TV}(P,Q)$, and $Q(W^c) = 1 - Q(W)$; adding the two gives the claim.

*(ii)* With $L_m := \prod_{i=1}^m \frac{dP}{dQ}(z_i)$ the likelihood ratio of the product, the representation of TV following (7.6) and Cauchy-Schwarz give
$$\mathrm{TV}\big(P^{\otimes m},Q^{\otimes m}\big) = \mathbb{E}_{Q^{\otimes m}}\big[(L_m-1)^+\big] \;\leq\; \Big(\mathbb{E}_{Q^{\otimes m}}\big[(L_m-1)^2\big]\Big)^{1/2} = \big(\chi^2(P^{\otimes m}\|Q^{\otimes m})\big)^{1/2}.$$
The chi-squared divergence of products **factorizes**, by independence of the $m$ coordinates and $\mathbb{E}_Q[L] = 1$:
$$1 + \chi^2\big(P^{\otimes m}\big\|Q^{\otimes m}\big) = \mathbb{E}_{Q^{\otimes m}}\big[L_m^2\big] = \prod_{i=1}^m \mathbb{E}_Q\big[L^2\big] = \big(1+\chi^2(P\|Q)\big)^m \;\leq\; e^{m\chi^2(P\|Q)},$$
and $e^x - 1 \leq 2x$ on $[0,1]$ gives $\chi^2(P^{\otimes m}\|Q^{\otimes m}) \leq 2m\,\chi^2(P\|Q)$.

*(iii)* Let $p,q$ be densities of $P,Q$ with respect to a common reference measure $\mu$ (for instance $\mu = P + Q$). Then $P(W) + Q(W^c) = \int_W p\,d\mu + \int_{W^c} q\,d\mu \geq \int \min(p,q)\,d\mu$. By Cauchy-Schwarz, and since $\int\max(p,q)\,d\mu = \int(p+q)\,d\mu - \int\min(p,q)\,d\mu \leq 2$,
$$\Big(\int\sqrt{pq}\,d\mu\Big)^2 = \Big(\int\sqrt{\min(p,q)\max(p,q)}\,d\mu\Big)^2 \;\leq\; \int\min(p,q)\,d\mu\cdot\int\max(p,q)\,d\mu \;\leq\; 2\int\min(p,q)\,d\mu.$$
By Jensen's inequality for the convex function $\exp$,
$$\begin{aligned}
\int\sqrt{pq}\,d\mu &\;\geq\; \int_{\{p>0\}} p\,\exp\Big(-\tfrac12\log\frac{p}{q}\Big)d\mu = \mathbb{E}_P\Big[\exp\Big(-\tfrac12\log\frac{dP}{dQ}\Big)\Big] \\
&\;\geq\; \exp\Big(-\tfrac12\mathrm{KL}(P\|Q)\Big).
\end{aligned}$$
Combining the three displays gives the inequality. For the product, $\log\frac{dP^{\otimes m}}{dQ^{\otimes m}}(z_1,\ldots,z_m) = \sum_i\log\frac{dP}{dQ}(z_i)$, whose expectation under $P^{\otimes m}$ is $m\,\mathrm{KL}(P\|Q)$ $\square$

Parts (i) and (ii) control the error probabilities when the two distributions are close in total variation; part (iii) remains informative when they are far apart, where it shows that the error probabilities decay no faster than $e^{-\mathrm{KL}}$. The first gives lower bounds on the expected error, the second on the sample size needed for a given confidence $\delta$.

---

**Theorem 7.9.3 (The biased-coin lower bound).** Let $\mathcal{X} = \{0\}$ be a one-point input space, $\mathcal{Y} = \{+1,-1\}$, and $\mathcal{H} = \{h_+, h_-\}$ the two constant hypotheses. Let $\mathcal{D}_+$ draw the label $+1$ with probability $\tfrac{1+\gamma}{2}$ and $-1$ otherwise, and let $\mathcal{D}_-$ be its mirror image. Let $A$ be any (possibly randomized) algorithm and $\gamma \in (0,\tfrac12]$.

*(i)* If $m \leq \frac{1}{64\gamma^2}$, then
$$\max_{j\in\{+,-\}}\ \mathbb{E}_{S\sim\mathcal{D}_j^m}\big[R_{\mathcal{D}_j}(A(S)) - R^*_{\mathcal{D}_j}\big] \;\geq\; \frac{\gamma}{4}.$$
*(ii)* For every $m \geq 1$,
$$\max_{j\in\{+,-\}}\ P_{S\sim\mathcal{D}_j^m}\big(R_{\mathcal{D}_j}(A(S)) - R^*_{\mathcal{D}_j} \geq \gamma\big) \;\geq\; \tfrac14\,e^{-4m\gamma^2}.$$

In words: an algorithm must tell a $(\tfrac12+\tfrac\gamma2)$-biased coin from a $(\tfrac12-\tfrac\gamma2)$-biased one, which requires $\Omega(1/\gamma^2)$ flips for constant confidence, and $\Omega(\log(1/\delta)/\gamma^2)$ flips for confidence $1-\delta$.

**Proof.** Since $\mathcal{X}$ is a single point, *every* hypothesis outputs one of the two labels there — the algorithm's entire decision is binary. Assume first that $A$ is deterministic, and let $W := \{S : A(S) = h_-\}$ be the samples on which it predicts $-1$. Under $\mathcal{D}_+$ the best hypothesis is $h_+$ and under $\mathcal{D}_-$ it is $h_-$; in both cases $R^* = \frac{1-\gamma}{2}$, and a hypothesis that errs under its distribution pays
$$R_{\mathcal{D}_+}(h_-) - R_{\mathcal{D}_+}(h_+) \;=\; R_{\mathcal{D}_-}(h_+) - R_{\mathcal{D}_-}(h_-) \;=\; \gamma.$$
So the excess risk is $\gamma$ on $W$ under $\mathcal{D}_+$, $\gamma$ on $W^c$ under $\mathcal{D}_-$, and $0$ otherwise.

*(i)* The expected excess risks are $\gamma\,\mathcal{D}_+^m(W)$ and $\gamma\,\mathcal{D}_-^m(W^c)$, so by Lemma 7.9.2(i),
$$\max_{\pm}\ \mathbb{E}\big[R_{\mathcal{D}_\pm}(A(S)) - R^*_{\mathcal{D}_\pm}\big] \;=\; \gamma\max\big(\mathcal{D}_+^m(W),\ \mathcal{D}_-^m(W^c)\big) \;\geq\; \frac{\gamma}{2}\big(1 - \mathrm{TV}(\mathcal{D}_+^m,\mathcal{D}_-^m)\big).$$
One label draw has chi-squared divergence
$$\chi^2(\mathcal{D}_+\|\mathcal{D}_-) \;=\; \sum_{y\in\{\pm1\}} \frac{\big(\mathcal{D}_+(y)-\mathcal{D}_-(y)\big)^2}{\mathcal{D}_-(y)} \;=\; \frac{2\gamma^2}{1-\gamma} + \frac{2\gamma^2}{1+\gamma} \;=\; \frac{4\gamma^2}{1-\gamma^2} \;\leq\; 6\gamma^2 \qquad (\gamma \leq \tfrac12),$$
since $\mathcal{D}_+(y) - \mathcal{D}_-(y) = \pm\gamma$. As $m \leq \frac{1}{64\gamma^2}$ gives $m\cdot 6\gamma^2 \leq \tfrac{6}{64} \leq 1$, Lemma 7.9.2(ii) applies:
$$\mathrm{TV}(\mathcal{D}_+^m,\mathcal{D}_-^m) \;\leq\; \sqrt{2m\cdot 6\gamma^2} \;\leq\; \sqrt{\tfrac{12}{64}} \;=\; \tfrac{\sqrt3}{4} \;<\; \tfrac12,$$
and the expected excess risk is at least $\frac{\gamma}{2}(1-\frac12) = \frac{\gamma}{4}$.

*(ii)* The two probabilities are $\mathcal{D}_+^m(W)$ and $\mathcal{D}_-^m(W^c)$; their maximum is at least half their sum, hence by Lemma 7.9.2(iii) at least $\tfrac14 e^{-m\,\mathrm{KL}(\mathcal{D}_+\|\mathcal{D}_-)}$. Here
$$\mathrm{KL}(\mathcal{D}_+\|\mathcal{D}_-) = \frac{1+\gamma}{2}\log\frac{1+\gamma}{1-\gamma} + \frac{1-\gamma}{2}\log\frac{1-\gamma}{1+\gamma} = \gamma\log\Big(1 + \frac{2\gamma}{1-\gamma}\Big) \;\leq\; \frac{2\gamma^2}{1-\gamma} \;\leq\; 4\gamma^2,$$
using $\log(1+u) \leq u$ and $\gamma \leq \tfrac12$.

Finally, a randomized $A$ is a deterministic function of the sample and an independent random variable $\omega \sim \mu$. The joint laws $\mathcal{D}_\pm^m \otimes \mu$ have the same likelihood ratio as $\mathcal{D}_\pm^m$, hence the same total variation and KL divergence, and both arguments apply verbatim to the pair $(S,\omega)$ $\square$

Part (i) rests on the factorization $1 + \chi^2(P^{\otimes m}\|Q^{\otimes m}) = (1+\chi^2(P\|Q))^m$ and the single-flip value $\chi^2 = 4\gamma^2/(1-\gamma^2)$: each sample improves distinguishability only by a factor $1 + O(\gamma^2)$. The separation $\gamma$ multiplies the error probability from the outside, while the total variation term only degrades it by a constant factor as long as $m \lesssim 1/\gamma^2$. Part (ii) converts into a sample-size requirement: for $\epsilon \in (0,\tfrac14)$ and $\delta \in (0,\tfrac14)$, take $\gamma = 2\epsilon$; an algorithm with $P(\text{excess risk} > \epsilon) \leq \delta$ under both coins needs $\tfrac14 e^{-16m\epsilon^2} \leq \delta$, i.e.
$$m \;\geq\; \frac{\log\big(1/(4\delta)\big)}{16\,\epsilon^2}. \tag{7.7}$$

The two-point method compares two distributions. Lower bounds that grow with the complexity of the class need many distributions. The classical tool for this is Fano's inequality, which lower-bounds the *average* error over a finite family through the **mutual information** between the index of the distribution and the sample. For $j$ uniform on $[N]$ and $S \sim \mathcal{D}_j^m$ given $j$, write $\bar{\mathcal{D}} := \frac1N\sum_{k=1}^N\mathcal{D}_k^m$ for the law of $S$, and define
$$I(j,S) := \mathrm{KL}\big(P_{(j,S)}\,\big\|\,P_j\otimes P_S\big) = \frac1N\sum_{j=1}^N\mathrm{KL}\big(\mathcal{D}_j^m\big\|\bar{\mathcal{D}}\big),$$
the KL divergence of the joint law of $(j,S)$ from the product of its marginals; the second expression follows since on the slice $\{j\}\times\mathcal{Z}^m$ the joint law is $\frac1N\mathcal{D}_j^m$ and the product law is $\frac1N\bar{\mathcal{D}}$, so that the likelihood ratio is $d\mathcal{D}_j^m/d\bar{\mathcal{D}}$. $I(j,S) = 0$ exactly when the sample carries no information about $j$.

**Theorem 7.9.4 (Fano's inequality, generalized).** Let $\mathcal{P} = \{\mathcal{D}_1,\ldots,\mathcal{D}_N\}$ be a finite family of distributions, $Q$ a quality measure as in Definition 7.9.1, $\epsilon > 0$, and
$$N_\epsilon := \sup_h\ \big|\{k : Q(h,\mathcal{D}_k) < \epsilon\}\big|,$$
the largest number of distributions a single hypothesis can serve at quality $\epsilon$. If $N_\epsilon < N$, then every (possibly randomized) algorithm $A$ satisfies
$$\frac1N\sum_{j=1}^N P_{S\sim\mathcal{D}_j^m}\big(Q(A(S),\mathcal{D}_j) < \epsilon\big) \;\leq\; \frac{I(j,S) + \log 2}{\log(N/N_\epsilon)}, \qquad I(j,S) \;\leq\; \frac{m}{N^2}\sum_{j,k=1}^N\mathrm{KL}(\mathcal{D}_j\|\mathcal{D}_k).$$

**Proof.** *Step 1: data processing onto an event.* For distributions $P \ll Q$ and any event $F$ with $0 < P(F) < 1$, with $\mathrm{kl}$ the binary KL divergence of Theorem 5.6.10,
$$\mathrm{KL}(P\|Q) \;\geq\; \mathrm{kl}\big(P(F)\,\big\|\,Q(F)\big).$$
Indeed, split $\mathrm{KL}(P\|Q) = P(F)\,\mathbb{E}_P[\log L \mid F] + P(F^c)\,\mathbb{E}_P[\log L \mid F^c]$ with $L := dP/dQ$. By Jensen's inequality for the concave $\log$, $\mathbb{E}_P[\log L \mid F] = -\mathbb{E}_P[\log(1/L) \mid F] \geq -\log\mathbb{E}_P[1/L \mid F] \geq -\log\frac{Q(F)}{P(F)}$, because $\mathbb{E}_P[\mathds{1}_F/L] = Q(F \cap \{L > 0\}) \leq Q(F)$; likewise on $F^c$. Adding the two bounds gives $P(F)\log\frac{P(F)}{Q(F)} + P(F^c)\log\frac{P(F^c)}{Q(F^c)}$. (If $P(F) \in \{0,1\}$ the same computation applies with the empty term dropped.)

*Step 2: apply it to the success event.* Take $P := P_{(j,S)}$, $Q := P_j\otimes P_S$ and $F := \{(j,S) : Q(A(S),\mathcal{D}_j) < \epsilon\}$. Then $P(F) =: \bar a$ is the average success probability on the left of the claim. Under $Q$, $j$ is uniform and independent of $S$, and for every fixed $S$ at most $N_\epsilon$ indices $j$ satisfy $Q(A(S),\mathcal{D}_j) < \epsilon$, so $Q(F) \leq N_\epsilon/N =: b$. Step 1 gives $I(j,S) \geq \mathrm{kl}(\bar a\,\|\,Q(F))$.

*Step 3: a lower bound on $\mathrm{kl}$.* For $a \in [0,1]$ and $c \in (0,1)$,
$$\mathrm{kl}(a\|c) = a\log\frac1c + (1-a)\log\frac{1}{1-c} + \big[a\log a + (1-a)\log(1-a)\big] \;\geq\; a\log\frac1c - \log 2,$$
since the middle term is non-negative and the bracket, a negative entropy of two outcomes, is at least $-\log 2$. With $c = Q(F) \leq b$, $\log\frac1c \geq \log\frac{N}{N_\epsilon}$; hence $I(j,S) \geq \bar a\log(N/N_\epsilon) - \log 2$, which rearranges to the first claim.

*Step 4: bound the mutual information.* Pointwise, by Jensen's inequality for the convex $-\log$,
$$\log\frac{d\mathcal{D}_j^m}{d\bar{\mathcal{D}}} = -\log\Big(\frac1N\sum_k\frac{d\mathcal{D}_k^m}{d\mathcal{D}_j^m}\Big) \;\leq\; \frac1N\sum_k\log\frac{d\mathcal{D}_j^m}{d\mathcal{D}_k^m}.$$ Taking $\mathbb{E}_{\mathcal{D}_j^m}$ and averaging over $j$ gives $I(j,S) \leq \frac{1}{N^2}\sum_{j,k}\mathrm{KL}(\mathcal{D}_j^m\|\mathcal{D}_k^m)$, and $\mathrm{KL}(\mathcal{D}_j^m\|\mathcal{D}_k^m) = m\,\mathrm{KL}(\mathcal{D}_j\|\mathcal{D}_k)$ by Lemma 7.9.2(iii). A randomized $A$ is handled by adjoining its internal randomness to $S$, which changes no KL divergence $\square$

Fano's inequality turns a lower bound into a *packing* problem: find many distributions that pairwise demand different answers ($N_\epsilon$ small) yet are hard to tell apart from $m$ samples ($I(j,S)$ small compared with $\log N$). The inequality goes back to Fano's information theory; versions with approximate recovery, as above, are developed in [Duchi & Wainwright (2013)](https://arxiv.org/abs/1311.2669)<!-- cite: duchi2013distance | misc | author={Duchi, John C. and Wainwright, Martin J.}; title={Distance-Based and Continuum Fano Inequalities with Applications to Statistical Estimation}; howpublished={arXiv:1311.2669}; year={2013} --> and in Chapter 12 of [Zhang (2023)](https://doi.org/10.1017/9781009093057)<!-- cite: zhang2023mathematical | book | author={Zhang, Tong}; title={Mathematical Analysis of Machine Learning Algorithms}; publisher={Cambridge University Press}; year={2023} -->; see also [Tsybakov (2009)](https://doi.org/10.1007/b13794)<!-- cite: tsybakov2009introduction | book | author={Tsybakov, Alexandre B.}; title={Introduction to Nonparametric Estimation}; publisher={Springer}; year={2009} -->. For classification over a shattered set the problem decouples into one binary test per point, and the two-point bounds of Lemma 7.9.2 applied coordinate by coordinate (*Assouad's method*, see Chapter 2 of Tsybakov, 2009) suffice and give simpler constants. The resulting lower bounds are classical; Chapter 28 of [Shalev-Shwartz & Ben-David (2014)](https://doi.org/10.1017/CBO9781107298019)<!-- cite: shalev2014understanding | book | author={Shalev-Shwartz, Shai and Ben-David, Shai}; title={Understanding Machine Learning: From Theory to Algorithms}; publisher={Cambridge University Press}; year={2014} --> proves the agnostic one with a different construction.

**Theorem 7.9.5 (Minimax lower bounds for VC classes).** Let $\mathcal{H}$ be a class of $\{0,1\}$-valued classifiers with $d_{VC}(\mathcal{H}) = d$, the loss being the zero-one loss, and let $A$ be any (possibly randomized) algorithm and $m \geq 1$.

*(i) Agnostic case ($d \geq 1$).* There is a distribution $\mathcal{D}$ with
$$P_{S\sim\mathcal{D}^m}\Big(R(A(S)) - \min_{h\in\mathcal{H}}R(h) \;\geq\; \tfrac{1}{64}\min\big(4,\sqrt{d/m}\big)\Big) \;\geq\; \tfrac17.$$
*(ii) Realizable case ($d \geq 2$).* There is a distribution $\mathcal{D}$ realizable by $\mathcal{H}$ with
$$P_{S\sim\mathcal{D}^m}\Big(R(A(S)) \;\geq\; \tfrac{1}{16}\min\big(2,(d-1)/m\big)\Big) \;\geq\; \tfrac17.$$
*(iii) Realizable case, confidence ($d \geq 2$).* For every $\epsilon \in (0,\tfrac14]$ there is a distribution $\mathcal{D}$ realizable by $\mathcal{H}$ with
$$P_{S\sim\mathcal{D}^m}\big(R(A(S)) > \epsilon\big) \;\geq\; \tfrac12\,e^{-4\epsilon m}.$$

**Proof.** Assume $A$ deterministic; randomized algorithms are handled as in Theorem 7.9.3. Each part follows the same pattern: a family of distributions indexed by labelings of a shattered set, an average over the family, and the elementary fact that a $[0,1]$-valued random variable $F$ with $\mathbb{E}[F] \geq \tfrac14$ satisfies $P(F \geq \tfrac18) \geq \tfrac17$, because $\mathbb{E}[F] \leq \tfrac18 + \tfrac78 P(F \geq \tfrac18)$.

*(i)* Let $C = \{x_1,\ldots,x_d\}$ be shattered by $\mathcal{H}$ and $\gamma := \min\big(\tfrac12,\sqrt{d/(64m)}\big)$, so that $m\gamma^2 \leq d/64$. For $\tau \in \{0,1\}^d$ let $\mathcal{D}_\tau$ draw $x$ uniformly from $C$ and, at $x = x_j$, the label $\tau_j$ with probability $\tfrac{1+\gamma}2$ and $1-\tau_j$ otherwise. The Bayes predictor $x_j \mapsto \tau_j$ belongs to $\mathcal{H}$ on $C$ (shattering), so $\min_{h}R_\tau(h) = R^*_{\mathcal{D}_\tau}$, and as in Theorem 7.9.3 every $h$ has excess risk
$$R_\tau(h) - \min_{h'\in\mathcal{H}}R_\tau(h') = \gamma\,F_\tau(h), \qquad F_\tau(h) := \frac1d\sum_{j=1}^d\mathds{1}\big(h(x_j) \neq \tau_j\big) \in [0,1].$$
Fix $j$ and pair each $\tau$ having $\tau_j = 1$ with $\tau' := \tau$ with coordinate $j$ flipped. With $W_j := \{S : A(S)(x_j) = 0\}$, the probabilities of erring at $x_j$ under $\mathcal{D}_\tau$ and $\mathcal{D}_{\tau'}$ are $\mathcal{D}_\tau^m(W_j)$ and $\mathcal{D}_{\tau'}^m(W_j^c)$, whose sum is at least $1 - \mathrm{TV}(\mathcal{D}_\tau^m,\mathcal{D}_{\tau'}^m)$ by Lemma 7.9.2(i). The two distributions differ only in the label law at $x_j$, which has mass $\tfrac1d$, so, as in Theorem 7.9.3,
$$\chi^2(\mathcal{D}_\tau\|\mathcal{D}_{\tau'}) = \frac1d\cdot\frac{4\gamma^2}{1-\gamma^2} \leq \frac{16\gamma^2}{3d}, \qquad m\,\chi^2 \leq \frac{16}{3}\cdot\frac{1}{64} = \frac1{12},$$
and Lemma 7.9.2(ii) gives $\mathrm{TV} \leq \sqrt{2/12} < \tfrac12$. Averaging over the pairs, for $\tau$ uniform on $\{0,1\}^d$, the probability of erring at $x_j$ is at least $\tfrac12(1-\tfrac12) = \tfrac14$, and averaging over $j$,
$$\mathbb{E}_\tau\,\mathbb{E}_{S\sim\mathcal{D}_\tau^m}\big[F_\tau(A(S))\big] \;\geq\; \tfrac14.$$
Some $\tau$ attains at least the average; for it $P(F_\tau(A(S)) \geq \tfrac18) \geq \tfrac17$, and on that event the excess risk is at least $\gamma/8 = \tfrac1{64}\min(4,\sqrt{d/m})$.

*(ii)* Let $C = \{x_0,x_1,\ldots,x_{d-1}\}$ be shattered, $p := \min\big(1,\frac{d-1}{2m}\big)$, and for $\tau \in \{0,1\}^{d-1}$ let $\mathcal{D}_\tau$ put mass $1-p$ on $x_0$ with label $0$ and mass $\frac{p}{d-1}$ on each $x_j$, $j \geq 1$, with label $\tau_j$. By shattering $\mathcal{D}_\tau$ is realizable. Let
$$R'_\tau := \frac{p}{d-1}\sum_{j=1}^{d-1}\mathds{1}\big(A(S)(x_j) \neq \tau_j\big) \;\leq\; R_\tau(A(S)), \qquad R'_\tau \in [0,p].$$
Fix $j \geq 1$, the input points of the sample, and all coordinates of $\tau$ except $\tau_j$. If $x_j$ does not occur in the sample, the sample does not depend on $\tau_j$, so for $\tau_j$ uniform $A(S)(x_j) \neq \tau_j$ with probability $\tfrac12$. The point $x_j$ is missed with probability $(1-\frac{p}{d-1})^m \geq 1 - \frac{mp}{d-1} \geq \tfrac12$. Hence $\mathbb{E}_\tau\mathbb{E}_S[R'_\tau] \geq \sum_{j\geq1}\frac{p}{d-1}\cdot\frac12\cdot\frac12 = \frac p4$; for a $\tau$ attaining the average, $F := R'_\tau/p$ has $\mathbb{E}[F] \geq \tfrac14$, so $R_\tau(A(S)) \geq R'_\tau \geq p/8 = \tfrac1{16}\min(2,\frac{d-1}{m})$ with probability at least $\tfrac17$.

*(iii)* Let $\{x_0,x_1\}$ be shattered and, for $\tau \in \{0,1\}$, let $\mathcal{D}_\tau$ put mass $1-2\epsilon$ on $x_0$ with label $0$ and mass $2\epsilon$ on $x_1$ with label $\tau$. With probability $(1-2\epsilon)^m \geq e^{-4\epsilon m}$ (by $\log(1-u) \geq -2u$ for $u \in [0,\tfrac12]$) the point $x_1$ does not occur in $S$, and then, for $\tau$ uniform, $A(S)(x_1) \neq \tau$ with probability $\tfrac12$, which makes $R_\tau(A(S)) \geq 2\epsilon > \epsilon$. Averaged over $\tau$ the probability of this is at least $\tfrac12 e^{-4\epsilon m}$, and some $\tau$ attains the average $\square$

Translated into sample complexities, for $\delta < \tfrac17$: part (i) shows that an agnostic learner with accuracy $\epsilon < \tfrac1{16}$ needs $\sqrt{d/m} \leq 64\epsilon$, i.e. $m \geq \frac{d}{4096\,\epsilon^2}$; part (ii) shows that a realizable learner with accuracy $\epsilon < \tfrac18$ needs $m \geq \frac{d-1}{16\epsilon}$; part (iii) adds $m \geq \frac{\log(1/(2\delta))}{4\epsilon}$ in the realizable case, and (7.7) adds $m \geq \frac{\log(1/(4\delta))}{16\epsilon^2}$ in the agnostic case (put $\mathcal{D}$ on a single point at which two hypotheses of $\mathcal{H}$ disagree). These are the lower bounds of the Fundamental Theorem (Theorem 6.4.7). The restriction $d \geq 2$ in (ii) and (iii) cannot be dropped: if $\mathcal{H} = \{h, 1-h\}$, a single example identifies the target, and the realizable sample complexity is $1$ for every $\epsilon$ and $\delta$.

---

What the minimax viewpoint adds to the upper bounds of this chapter is a **certificate of optimality**. The chaining bound (7.4) shows that ERM achieves excess risk $O(\sqrt{d/m})$ on every distribution; Theorem 7.9.5 shows that no algorithm does better in the worst case, and Theorem 7.9.3 that the confidence term $\log(1/\delta)$ is necessary as well. The rates proved in this chapter are therefore properties of the *learning problem*, not artifacts of the proof technique.

# Further Directions

Three lines of work continue the theory of this chapter.

* **Fast rates under low noise.** Apart from the realizable case (Theorem 6.4.14) and the stability-based oracle inequality after Theorem 7.8.3, which pays for its $O(1/m)$ rate with an expectation over $S$ and a comparison against the regularized objective, all bounds above scale as $O(1/\sqrt m)$. Under a **low-noise (margin) condition** — informally, when $\mathcal{D}$ rarely places points close to the decision boundary — the excess risk of ERM decays faster, interpolating between $1/\sqrt m$ and $1/m$ as the condition strengthens. The mechanism is the one behind Theorem 6.4.14: when the variance of the excess loss is controlled by its mean, Bernstein's inequality (Theorem 5.6.9) replaces Hoeffding's, and localized versions of the Rademacher complexity replace the global one. The margin bound for boosting (Theorem 12.2.7) is a related instance.
* **Sharper lower bounds.** Theorem 7.9.5 fixes the rates but not the constants. Exact minimax constants for specific problem families, and lower bounds for estimation problems with continuous parameters, are obtained by sharper packing constructions combined with Fano's inequality and Assouad's method; see [Tsybakov (2009)](https://doi.org/10.1007/b13794)<!-- cite: tsybakov2009introduction | book | author={Tsybakov, Alexandre B.}; title={Introduction to Nonparametric Estimation}; publisher={Springer}; year={2009} -->.
* **Generic chaining.** Dudley's entropy integral (Theorem 7.6.2) is not always tight. Its refinement, generic chaining, characterizes the expected supremum of a Gaussian process up to universal constants (the majorizing-measure theorem), and with it the Rademacher complexity of many classes; see Chapter 12 of [Zhang (2023)](https://doi.org/10.1017/9781009093057)<!-- cite: zhang2023mathematical | book | author={Zhang, Tong}; title={Mathematical Analysis of Machine Learning Algorithms}; publisher={Cambridge University Press}; year={2023} -->.
