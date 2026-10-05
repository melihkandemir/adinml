This chapter builds the probabilistic machinery the rest of the course runs on. The notational conventions it uses were fixed in Chapter 1 and are assumed throughout.

# Measure-Theoretic Foundations

**Sample Space ($\Omega$):** The set of all possible outcomes of a random process.

**Event ($A$):** A subset of the sample space, $A \subset \Omega$.

$2^\Omega$ is the **powerset** of $\Omega$. Let $\mathcal{F} \subseteq 2^\Omega$ be the collection of events we are willing to assign a probability to. Then $\mathcal{F}$ is called the **event space**. Technically speaking, it is a **$\sigma$-algebra**: a collection of subsets of $\Omega$ that contains $\Omega$ itself and is closed under complementation and countable unions. That is, $\Omega \in \mathcal{F}$, and if $A_1, A_2, \dots \in \mathcal{F}$, then:

1. $A_1^c \in \mathcal{F}$,
2. $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$.

Closure under countable intersections, $\bigcap_{i=1}^\infty A_i \in \mathcal{F}$, follows from these two axioms via De Morgan's law: $\bigcap_i A_i = \big(\bigcup_i A_i^c\big)^c$.

**Why we need this at all.** For discrete sample spaces, such as the outcome of the roll of a pair of dice, it is safe to take $\mathcal{F} = 2^\Omega$: every subset is an event, and we can assign a probability to it. For continuous sample spaces, such as $\Omega = \mathbb{R}$, the powerset $2^{\mathbb{R}}$ is too large — it contains pathological sets to which no consistent notion of "length" (and hence probability) can be assigned. The fix is to only ever ask for the probability of sets built from intervals. The **Borel $\sigma$-algebra** $\mathcal{B}(\mathbb{R})$ is the smallest $\sigma$-algebra containing all open intervals $(a,b) \subset \mathbb{R}$; it contains every set we will ever need in practice (all open sets, closed sets, countable unions and intersections thereof) while excluding the pathological ones. When $\Omega = \mathbb{R}^d$, we use $\mathcal{F} = \mathcal{B}(\mathbb{R}^d)$, the $\sigma$-algebra generated analogously by open rectangles.

A **probability measure** is a function $P: \mathcal{F} \rightarrow [0,1]$ such that:

1. $P(\Omega) = 1$,
2. $P(A) \geq 0$ for all $A \in \mathcal{F}$,
3. **($\sigma$-additivity)** if $A_1, A_2, \dots$ are pairwise disjoint events (i.e. $A_i \cap A_j = \emptyset$ for all $i \neq j$), then $P\big(\bigcup_{i=1}^{\infty} A_i\big) = \sum_{i=1}^{\infty} P(A_i)$.

**Probability Space:** The triple $(\Omega, \mathcal{F}, P)$. This is the complete mathematical object a random experiment is modeled by. Everything else in this chapter — random variables, expectations, concentration inequalities — is built on top of it.

**Intuitive meaning of probability.** While $P$ is defined axiomatically above, the frequentist intuition behind it is the long-run frequency of an event across repeated, independent trials:

$$P(A) = \lim_{n \rightarrow \infty} \frac{n_A}{n},$$

where $n_A$ is the number of times event $A$ occurs in $n$ trials. This intuition is useful for building understanding, but it is not the definition; the axioms above are.

# Basic Rules

The rules below follow directly from the axioms and are standard fare; we state them compactly since they will be used freely throughout the rest of the notes without re-derivation.

* **Union bound.** From the inclusion-exclusion identity $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ and $P(A \cap B) \geq 0$, we get $P(A \cup B) \leq P(A) + P(B)$. By induction, for any (not necessarily disjoint) events $A_1, \ldots, A_n$,
$$P\Big(\bigcup_{i=1}^n A_i\Big) \leq \sum_{i=1}^n P(A_i).$$
This inequality — despite its triviality — is the single most used tool in this course: it lets us bound the probability that *any one* of many bad events happens by summing the probabilities of each bad event individually.

* **Conditional probability.** $P(A \mid B) := \dfrac{P(A \cap B)}{P(B)}$ for $P(B) > 0$, read as "the probability of $A$ given that $B$ has occurred." Rearranged, this gives the **product rule**: $P(A \cap B) = P(A \mid B) P(B)$.

* **Independence.** Events $A, B$ are **independent** if $P(A \mid B) = P(A)$, equivalently $P(A \cap B) = P(A) P(B)$.

* **Law of total probability.** If $B_1, \ldots, B_n$ partition $\Omega$ (pairwise disjoint, $\bigcup_i B_i = \Omega$), then $P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)$.

* **Bayes' rule** [Bayes and Price (1763)](https://doi.org/10.1098/rstl.1763.0053)<!-- cite: bayes1763essay | article | author={Bayes, Thomas and Price, Richard}; title={An Essay towards Solving a Problem in the Doctrine of Chances}; journal={Philosophical Transactions of the Royal Society of London}; year={1763}; volume={53}; pages={370--418} -->. From the product rule applied both ways, $P(A \mid B) P(B) = P(A \cap B) = P(B \mid A) P(A)$, hence
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

While each of these facts is elementary on its own, Bayes' rule has outsized implications for statistical inference and machine learning. Assume $H_1, \ldots, H_n$ are mutually exclusive hypotheses partitioning a hypothesis space $\mathcal{H}$, and $\mathcal{D}$ is observed data. Combining Bayes' rule with the law of total probability (using $H_1,\ldots,H_n$ as the partition) gives

$$P(H_i \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid H_i) P(H_i)}{P(\mathcal{D})} = \frac{P(\mathcal{D} \mid H_i) P(H_i)}{\sum_{j=1}^n P(\mathcal{D} \mid H_j) P(H_j)}.$$

Here:

* $P(H_i)$ is our **prior belief** in hypothesis $H_i$,
* $P(H_i \mid \mathcal{D})$ is our **posterior belief** in $H_i$ after observing the data $\mathcal{D}$,
* $P(\mathcal{D} \mid H_i)$ is the **likelihood** of $H_i$, and $P(\mathcal{D})$ is the **evidence**.

This gives a general recipe for statistical inference: start with a prior belief, collect observations, update the belief. Chapter 6 builds an entire class of learning algorithms on exactly this recipe.

# Random Variables

The probability measure $P$ maps *sets* to numbers. It is not always convenient to work directly with sets; we would rather describe outcomes numerically and delegate the set-theoretic bookkeeping to the machinery below. This is what a random variable does.

**Definition (Random variable).** Let $(\Omega, \mathcal{F}, P)$ be a probability space and $(\Lambda, \mathcal{G})$ a measurable space (e.g. $\Lambda = \mathbb{R}^k$ with its Borel $\sigma$-algebra $\mathcal{G} = \mathcal{B}(\mathbb{R}^k)$). A **random variable** is a function $X: \Omega \rightarrow \Lambda$ that is **measurable**, meaning
$$\begin{gathered}
X^{-1}(A) := \{\omega \in \Omega : X(\omega) \in A\} \in \mathcal{F}, \\
\forall A \in \mathcal{G}.
\end{gathered}$$

---

Measurability is exactly the condition needed for the expression "$P(X \in A)$" to make sense: it guarantees that the set of outcomes $\omega$ mapping into $A$ is itself an event, i.e. something $P$ can be evaluated on. Given this, we define the **induced probability measure** (a.k.a. the **law** or **distribution**) of $X$ as
$$\begin{gathered}
P_X(A) := P(X^{-1}(A)) = P(\{\omega \in \Omega: X(\omega) \in A\}), \\
A \in \mathcal{G}.
\end{gathered}$$

$(\Lambda, \mathcal{G}, P_X)$ is itself a probability space. This is the sense in which a random variable "transports" a probability space we may not care about the internals of (e.g. the space of all coin-toss sequences) onto a space we do care about (e.g. $\{0,1,\ldots,n\}$, the count of heads).

**Discrete random variables.** When $\Lambda$ is countable, e.g. $\Lambda = \{0,1,\ldots,C-1\}$, every subset of $\Lambda$ is measurable and it suffices to specify $P_X$ on singletons:
$$\begin{gathered}
P(X = x) := P_X(\{x\}) = P(\{\omega \in \Omega: X(\omega) = x\}), \\
x \in \Lambda.
\end{gathered}$$
This is called the **probability mass function (pmf)** of $X$, and $P_X(A) = \sum_{x \in A} P(X=x)$ recovers the measure on any event $A$. Note that we write a pmf with a **capital** $P$, because it *is* the probability of the event $\{X=x\}$ — no new object is being introduced. Lowercase $p$ is reserved, throughout these notes, for the densities of continuous random variables defined next, which are **not** probabilities of anything.

**Continuous random variables.** When $\Lambda = \mathbb{R}^k$, singletons typically carry zero probability ($P_X(\{x\}) = 0$), so a pmf is useless; instead we ask for $P_X$ to be absolutely continuous with respect to the Lebesgue measure $\mu_{\text{Leb}}$ (ordinary $k$-dimensional volume). This means there exists a function $p_X \geq 0$, called the **probability density function (pdf)**, such that
$$\begin{gathered}
P_X(A) = \int_A p_X(x) \, d\mu_{\text{Leb}}(x), \\
\forall A \in \mathcal{B}(\mathbb{R}^k).
\end{gathered}$$
For $k=1$ this is the familiar $P_X(A) = \int_A p_X(x)\,dx$. Not every random variable admits a density (some are discrete, some are neither), but every case relevant to this course is either discrete or admits one.

**Example (three coin tosses).** Toss a fair coin three times; write $H$ for heads and $T$ for tails. The sample space is

$$\Omega = \{HHH, HHT, HTH, HTT, THH, THT, TTH, TTT\} ,$$

and since the coin is fair and the tosses are independent, $P(\omega) = \big(\tfrac12\big)^3 = \tfrac18$ for every $\omega \in \Omega$. Let $X: \Omega \to \{0,1,2,3\}$ count the number of heads. Then $X$ is a (measurable, since $\Omega$ is finite and $\mathcal{F}=2^\Omega$) random variable with pmf

1. $P(X=0) = P(\{TTT\}) = \frac{1}{8}$,
2. $P(X=1) = P(\{TTH, THT, HTT\}) = \frac{3}{8}$,
3. $P(X=2) = P(\{HTH, HHT, THH\}) = \frac{3}{8}$,
4. $P(X=3) = P(\{HHH\}) = \frac{1}{8}$.

---

Random variables also let us query compound events easily, e.g.
$$P(X \geq 2) = P(\{HTH, HHT, THH, HHH\}) = \frac{1}{2}.$$

**Joint distributions, marginals, independence.** Given two random variables $X, Y$ on the same probability space, their **joint distribution** is $P(X=x, Y=y) := P(\{\omega : X(\omega)=x, Y(\omega)=y\})$ (with the analogous density definition in the continuous case). We say $X$ and $Y$ are **independent** if $P(X=x, Y=y) = P(X=x)P(Y=y)$ for all $x,y$ — the random-variable counterpart of event independence above.

When a variable in a joint distribution is not of direct interest, it is called a **nuisance variable**, and we can eliminate it by the law of total probability. This operation is called **marginalization**. For instance, if $X$ is the outcome of a fair die roll and $Y \in \{\text{even},\text{odd}\}$ its parity,
$$
\begin{aligned}
P(X=x) &= \sum_{y \in \{\text{even},\text{odd}\}} P(X=x, Y=y) \\
&= \sum_{y} P(X=x \mid Y=y) P(Y=y).
\end{aligned}
$$
Using $P(X=x)$ lets us query events concerning $X$ alone, with $Y$'s influence already averaged out proportionally to its own probability of occurrence. (As a sanity check on the definition of independence above: for a fair die, $X \leq 4$ and $Y=\text{even}$ are independent — $P(X\le 4, Y{=}\text{even}) = \tfrac{2}{6}$ equals $P(X\le4)P(Y{=}\text{even}) = \tfrac46 \cdot \tfrac36$ — while $X \leq 3$ and $Y=\text{even}$ are not, since low rolls are disproportionately odd.)

# Expectation as an Integral

**Definition (Expectation).** The **expectation** of a random variable $X: \Omega \to \mathbb{R}$ is the Lebesgue integral of $X$ against $P$:
$$\mathbb{E}[X] := \int_\Omega X(\omega) \, dP(\omega).$$

---

This single definition specializes to the two formulas you have likely seen before, depending on the law of $X$:

* if $X$ is discrete with pmf $P(X=\cdot)$, the abstract integral reduces to a sum: $\mathbb{E}[X] = \sum_{x \in \Lambda} x \, P(X=x)$;
* if $X$ is continuous with pdf $p_X$, it reduces to a Riemann-style integral: $\mathbb{E}[X] = \int_{\mathbb{R}} x \, p_X(x) \, dx$.

Framing expectation as a single integral against $P$, rather than as two unrelated formulas, is precisely the payoff of the measure-theoretic viewpoint: every fact we prove about $\mathbb{E}[\cdot]$ below (linearity, monotonicity) holds simultaneously for discrete and continuous random variables — and for mixtures of the two — because it is proven once, at the level of the integral, rather than twice.

In the three coin tosses example above, $\mathbb{E}[X] = 0 \cdot \frac18 + 1 \cdot \frac38 + 2 \cdot \frac38 + 3 \cdot \frac18 = 1.5$.

The expectation is also called the **mean** or **first moment** of $X$; it summarizes the **central tendency** of $X$ with a single number, but says nothing about its **spread**. Spread is measured by the **variance**:
$$\mathrm{Var}[X] := \mathbb{E} \big[(X-\mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - \mathbb{E}[X]^2.$$
In the example above, $\mathrm{Var}[X] = 0^2\cdot\frac18 + 1^2\cdot\frac38 + 2^2\cdot\frac38 + 3^2\cdot\frac18 - 1.5^2 = 0.75$. The square root of the variance is the **standard deviation**, commonly denoted $\sigma$; here $\sigma = \sqrt{0.75} \approx 0.866$.

Variance is also called the **second (central) moment** of $X$. In general, the **$k$-th central moment** is $\mathbb{E}[(X-\mathbb{E}[X])^k]$ for a positive integer $k$: the first is (trivially) zero and the second is the variance. The third and fourth become scale-free, and acquire their usual names, only after being standardized by the appropriate power of $\sigma$: the **skewness** $\mathbb{E}[(X-\mathbb{E}[X])^3]/\sigma^3$ measures the asymmetry of the distribution, and the **kurtosis** $\mathbb{E}[(X-\mathbb{E}[X])^4]/\sigma^4$ the heaviness of its tails.

For two jointly distributed random variables $X, Y$, the **covariance** quantifies how their deviations from their own means co-vary:
$$\mathrm{Cov}[X,Y] := \mathbb{E} \big[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\big] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y].$$
Setting $Y=X$ recovers the variance, $\mathrm{Cov}[X,X] = \mathrm{Var}[X]$.

# Estimators and Bias

**Estimator.** A function that maps a sample $S$ to a value $\widehat{\theta}(S)$ is called an **estimator** of a parameter $\theta$. We write the estimator and the value it produces with the same symbol $\widehat{\theta}$, suppressing the argument $S$ when it is clear from context. The running example throughout this chapter is the **sample mean** of an i.i.d. sample $X_1,\ldots,X_m \sim X$,
$$\widehat{\mu}_m := \frac{1}{m}\sum_{i=1}^m X_i,$$
which estimates the mean $\mu := \mathbb{E}[X]$.

**Bias.** The bias of an estimator is $\mathrm{Bias}(\widehat\theta) := \mathbb{E}[\widehat\theta(S)] - \theta$. An estimator is **unbiased** if $\mathrm{Bias}(\widehat\theta)=0$. For example, $\widehat{\mu}_m$ is unbiased for $\mu$:
$$
\begin{aligned}
\mathrm{Bias}(\widehat{\mu}_m) &= \mathbb{E}[\widehat{\mu}_m] - \mu = \mathbb{E} \Big[\frac{1}{m}\sum_{i=1}^m X_i\Big] - \mu = \frac{1}{m}\sum_{i=1}^m \mathbb{E}[X_i] - \mu = \frac{1}{m}\sum_{i=1}^m \mu - \mu = 0.
\end{aligned}
$$
By contrast, the sample variance $\widehat{\sigma}_m^2 := \frac{1}{m}\sum_{i=1}^m (X_i - \widehat{\mu}_m)^2$ is a *biased* estimator of $\sigma^2$; the **unbiased sample variance** $\frac{1}{m-1}\sum_{i=1}^m (X_i-\widehat{\mu}_m)^2$ corrects for this via **Bessel's correction**. (We reserve $S$ for a data set throughout these notes, which is why the sample variance is not written $S^2$ as it often is in statistics texts.)

**Consistency.** An estimator is **consistent** if $\lim_{m\to\infty} P(|\widehat\theta(S)-\theta| \geq \epsilon) = 0$ for every $\epsilon > 0$: the probability of a large error vanishes as the sample grows. The next section builds the tools to establish (and quantify) consistency.

# Concentration of Measure Inequalities

**Theorem 4.1 (Markov's inequality).** Let $X$ be a non-negative random variable. Then for any $\epsilon > 0$,
$$P(X \geq \epsilon) \leq \frac{\mathbb{E}[X]}{\epsilon}.$$

**Proof.** Since $X \geq 0$, the pointwise inequality $X \geq \epsilon\, \mathds{1}(X \geq \epsilon)$ holds for every outcome: if $X < \epsilon$ the right side is $0 \leq X$, and if $X \geq \epsilon$ the right side is $\epsilon \leq X$. Taking expectations and using monotonicity and linearity of the integral,
$$\mathbb{E}[X] \geq \epsilon\, \mathbb{E}\big[\mathds{1}(X \geq \epsilon)\big] = \epsilon\, P(X \geq \epsilon),$$
where the last equality is the defining property of the indicator: $\mathbb{E}[\mathds{1}(A)] = P(A)$. Dividing by $\epsilon > 0$ gives the claim $\square$

Note how little the proof uses: only non-negativity and linearity. This is why Markov's inequality is the weakest of the three bounds in this section — and why every sharper bound below is obtained by first *transforming* $X$ into a quantity to which Markov can be applied more profitably.

**Example.** Consider ten Bernoulli random variables $X_1,\ldots,X_{10}$ with $P(X_i=1)=P(X_i=0)=0.5$. Since $\mathbb{E}[\sum_{i=1}^{10}X_i]=5$, Markov's inequality only gives $P(\sum_{i=1}^{10}X_i \geq 3) \leq 5/3$, which exceeds 1 and is thus **vacuous**.

**Theorem 4.2 (Chebyshev's inequality).** Let $X$ have finite mean $\mu$ and variance $\sigma^2$. Then for any $\epsilon > 0$,
$$P(|X-\mu| \geq \epsilon) \leq \frac{\sigma^2}{\epsilon^2}.$$

**Proof.** The events $\{|X-\mu| \geq \epsilon\}$ and $\{(X-\mu)^2 \geq \epsilon^2\}$ are identical, since $u \mapsto u^2$ is increasing on $[0,\infty)$. The random variable $(X-\mu)^2$ is non-negative, so Theorem 4.1 applies to it with threshold $\epsilon^2$:
$$P(|X-\mu| \geq \epsilon) = P\big((X-\mu)^2 \geq \epsilon^2\big) \leq \frac{\mathbb{E}[(X-\mu)^2]}{\epsilon^2} = \frac{\mathrm{Var}[X]}{\epsilon^2} = \frac{\sigma^2}{\epsilon^2}. \qquad\square$$

This is the transformation trick announced above, in its simplest form: squaring turns a two-sided deviation into a non-negative quantity, and buys a factor $1/\epsilon$ over Markov by paying with the assumption of a finite second moment. Applying Markov to $e^{sX}$ instead of $(X-\mu)^2$ — the **Chernoff trick** — buys an *exponential* improvement, at the price of assuming a finite moment generating function; this is exactly the route Theorem 4.4 takes below.

Consider $S = \{X_1,\ldots,X_m\}$ i.i.d. Bernoulli with sample average $\widehat{\mu}_m$ as defined above. How much can $\widehat\mu_m$ deviate from $\mu$? Markov's inequality alone gives us

$$
\begin{aligned}
P(\widehat{\mu}_m - \mathbb{E}[\widehat{\mu}_m] \geq \epsilon) &= P(\widehat{\mu}_m \geq \mu + \epsilon) \leq \frac{\mathbb{E}[\widehat{\mu}_m]}{\mu+\epsilon} = \frac{\mu}{\mu+\epsilon},
\end{aligned}
$$

a bound that does not shrink no matter how large $m$ is — useless for arguing that more data helps. Chebyshev's inequality does better:

$$
\begin{aligned}
P(|\widehat{\mu}_m - \mathbb{E}[\widehat{\mu}_m]| \geq \epsilon) &\leq \frac{\mathrm{Var}[\widehat{\mu}_m]}{\epsilon^2} = \frac{\mathrm{Var}\big[\frac{1}{m}\sum_{i=1}^m X_i\big]}{\epsilon^2} = \frac{1}{m^2}\frac{\sum_{i=1}^m \mathrm{Var}[X_i]}{\epsilon^2} \\
&= \frac{1}{m^2}\frac{m\,\mathrm{Var}[X_i]}{\epsilon^2} = \frac{\sigma^2}{m\epsilon^2},
\end{aligned}
$$

using $\mathrm{Var}[\sum_i X_i] = \sum_i \mathrm{Var}[X_i]$ for independent $X_i$ (second step) and $\mathrm{Var}[aX]=a^2\mathrm{Var}[X]$ for a constant $a$ (third step). This bound shrinks as $\Theta(1/m)$, giving the **weak law of large numbers**:

**Theorem 4.3 (Weak law of large numbers).** Let $X_1,\ldots,X_m$ be i.i.d. with finite mean $\mu$ and variance $\sigma^2$. Then for any $\epsilon > 0$, $\lim_{m\to\infty} P(|\widehat{\mu}_m - \mu| \geq \epsilon) = 0$, where $\widehat{\mu}_m = \frac{1}{m}\sum_{i=1}^m X_i$.

**Proof.** $\widehat\mu_m$ is unbiased, $\mathbb{E}[\widehat\mu_m] = \mu$, and the display preceding this theorem computed $\mathrm{Var}[\widehat\mu_m] = \sigma^2/m$. Chebyshev's inequality (Theorem 4.2) applied to $\widehat\mu_m$ therefore gives
$$0 \leq P\big(|\widehat\mu_m - \mu| \geq \epsilon\big) \leq \frac{\sigma^2}{m\epsilon^2} \xrightarrow[m\to\infty]{} 0$$
for every fixed $\epsilon > 0$, and the claim follows by squeezing $\square$

Hence $\widehat\mu_m$ is a consistent estimator of $\mu$. Note what the proof needed and what it did not: finiteness of $\sigma^2$ was used, but no boundedness and no distributional shape. (Dropping the finite-variance assumption still leaves the theorem true, by a truncation argument we do not need here.)

## Hoeffding's inequality

Chebyshev's inequality gets tighter at the polynomial rate $1/m$. Boundedness lets us do much better — exponentially in $m$, with a bound due to [Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830)<!-- cite: hoeffding1963probability | article | author={Hoeffding, Wassily}; title={Probability Inequalities for Sums of Bounded Random Variables}; journal={Journal of the American Statistical Association}; year={1963}; volume={58}; number={301}; pages={13--30} --> that we derive below.

The device is the **Chernoff trick**: instead of applying Markov's inequality to $X$ or to $(X-\mu)^2$, apply it to $e^{sX}$ for a free parameter $s>0$, then optimize over $s$. What makes this profitable is that the exponential turns a sum of independent variables into a *product* of expectations, and that each factor can be bounded using boundedness alone. The latter is the content of the following lemma.

**Lemma 4.1 (Hoeffding's lemma).** Let $X$ be a random variable with $\mathbb{E}[X]=0$ and $P(a \leq X \leq b)=1$. Then for every $s \in \mathbb{R}$,
$$\mathbb{E}\big[e^{sX}\big] \leq \exp\Big(\frac{s^2(b-a)^2}{8}\Big).$$

**Proof.** Note $a \leq 0 \leq b$, since $\mathbb{E}[X]=0$. Because $u \mapsto e^{su}$ is convex, every $x \in [a,b]$, written as the convex combination $x = \lambda b + (1-\lambda) a$ with $\lambda = \frac{x-a}{b-a}$, satisfies
$$e^{sx} \leq \frac{x-a}{b-a}e^{sb} + \frac{b-x}{b-a}e^{sa}.$$
Taking expectations and using $\mathbb{E}[X]=0$ to kill the $x$-dependent parts,
$$\begin{gathered}
\mathbb{E}\big[e^{sX}\big] \leq \frac{-a}{b-a}e^{sb} + \frac{b}{b-a}e^{sa} = p\,e^{sb} + (1-p)e^{sa}, \\
p := \frac{-a}{b-a} \in [0,1].
\end{gathered}$$
Substitute $u := s(b-a)$, so that $sa = -pu$, and factor out $e^{sa}$:
$$\begin{gathered}
p\,e^{sb} + (1-p)e^{sa} = e^{-pu}\big(1 - p + p\,e^{u}\big) = e^{\varphi(u)}, \\
\varphi(u) := -pu + \log\big(1-p+p e^{u}\big).
\end{gathered}$$
It remains to show $\varphi(u) \leq u^2/8$. Differentiating,
$$\begin{gathered}
\varphi'(u) = -p + \frac{p e^u}{1-p+pe^u}, \\
\varphi''(u) = q(u)\big(1-q(u)\big) \ \text{ with }\ q(u) := \frac{pe^u}{1-p+pe^u} \in [0,1].
\end{gathered}$$
Hence $\varphi(0)=0$, $\varphi'(0)=0$, and $\varphi''(u) \leq \tfrac14$ everywhere, since $q(1-q) \leq \tfrac14$ for $q \in [0,1]$. Taylor's theorem with Lagrange remainder around $0$ therefore gives $\varphi(u) = \tfrac12\varphi''(\xi)u^2 \leq u^2/8$ for some $\xi$ between $0$ and $u$. Substituting back $u = s(b-a)$ completes the proof $\square$

**Theorem 4.4 (Hoeffding's inequality).** Let $X_1,\ldots,X_m$ be independent random variables such that $P(a_i \leq X_i \leq b_i)=1$ for all $i$. Then for any $\epsilon>0$,
$$P\Big(\frac{1}{m}\sum_{i=1}^m X_i - \mathbb{E} \Big[\frac{1}{m}\sum_{i=1}^m X_i\Big] \geq \epsilon\Big) \leq \exp\Big(\frac{-2m^2\epsilon^2}{\sum_{i=1}^m(b_i-a_i)^2}\Big).$$

**Proof.** Center the variables, $Y_i := X_i - \mathbb{E}[X_i]$, so that $\mathbb{E}[Y_i]=0$ and $Y_i$ lies in an interval of the same width $b_i-a_i$. For any $s>0$, the map $u \mapsto e^{su}$ is increasing, so Markov's inequality (Theorem 4.1) applied to the non-negative variable $e^{s\sum_i Y_i}$ gives

$$
\begin{aligned}
P\Big(\sum_{i=1}^m Y_i \geq m\epsilon\Big) = P\Big(e^{s\sum_i Y_i} \geq e^{sm\epsilon}\Big) &\leq e^{-sm\epsilon}\, \mathbb{E}\Big[e^{s\sum_i Y_i}\Big] \\
&= e^{-sm\epsilon} \prod_{i=1}^m \mathbb{E}\big[e^{sY_i}\big] \\
&\leq e^{-sm\epsilon} \exp\Big(\frac{s^2}{8}\sum_{i=1}^m (b_i-a_i)^2\Big),
\end{aligned}
$$

where the equality on the second line uses independence (the expectation of a product of independent variables factorizes) and the last step applies Lemma 4.1 to each $Y_i$ separately. The bound holds for every $s>0$, so we may take the best one. Writing $V := \sum_i (b_i-a_i)^2$, the exponent $-sm\epsilon + s^2 V/8$ is a convex quadratic in $s$, minimized at $s^* = 4m\epsilon/V > 0$, where its value is

$$-\frac{4m^2\epsilon^2}{V} + \frac{2m^2\epsilon^2}{V} = -\frac{2m^2\epsilon^2}{V}.$$

Substituting and dividing the event by $m$ gives the claim $\square$

Applying Theorem 4.4 to the variables $-X_1,\ldots,-X_m$ (which lie in $[-b_i,-a_i]$, of the same width) bounds the opposite tail by the same quantity:
$$P\Big(\mathbb{E} \Big[\frac{1}{m}\sum_{i=1}^m X_i\Big] - \frac{1}{m}\sum_{i=1}^m X_i \geq \epsilon\Big) \leq \exp\Big(\frac{-2m^2\epsilon^2}{\sum_{i=1}^m(b_i-a_i)^2}\Big),$$
and adding the two gives the two-sided version with an extra factor of $2$.

We will use these inequalities to bound the generalization performance of learning algorithms. Let $X_i$ be the loss of the $i$-th training example; then $\widehat\mu_m = \frac{1}{m}\sum_{i=1}^m X_i$ is the empirical risk and $\mu = \mathbb{E}[\widehat\mu_m]$ the expected risk. It is the lower tail that we need — the event that the true risk exceeds what training led us to believe. If $X_i \in [0,1]$ for all $i$ (so $b_i - a_i = 1$), it simplifies to
$$P(\mu \geq \widehat\mu_m + \epsilon) \leq e^{-2m\epsilon^2}.$$
Setting $\delta := e^{-2m\epsilon^2}$ and solving for $\epsilon$ gives $\epsilon = \sqrt{\log(1/\delta)/(2m)}$, hence
$$\begin{gathered}
P\Big(\mu \geq \widehat\mu_m + \sqrt{\tfrac{\log(1/\delta)}{2m}}\Big) \leq \delta, \\
\text{equivalently}\quad P\Big(\mu \leq \widehat\mu_m + \sqrt{\tfrac{\log(1/\delta)}{2m}}\Big) \geq 1-\delta.
\end{gathered}$$
Since we will later allocate $\mu$ to generalization performance and $\widehat\mu_m$ to training performance, it is this second, complementary form that we will reuse directly. Concentration inequalities of this form are called **generalization bounds**, and are the basis of statistical learning theory (Chapter 5).

## Uniform Hoeffding bounds over several estimators

A single Hoeffding bound controls the deviation of *one* empirical average from its mean. In machine learning we rarely evaluate a single quantity: we compare many hypotheses' training errors, or track several estimators built from ranges (loss scales) that differ from one another. We now extend Theorem 4.4 to this setting, using nothing more than the union bound.

**Setup.** Suppose we have $K$ estimators $\widehat\mu_m^{(1)}, \ldots, \widehat\mu_m^{(K)}$, where $\widehat\mu_m^{(k)} := \frac{1}{m}\sum_{i=1}^m X_i^{(k)}$ is built from $m$ independent observations $X_1^{(k)},\ldots,X_m^{(k)}$ (independent across $i$; the $K$ estimators themselves need not be independent of each other) with each $X_i^{(k)} \in [a_k, b_k]$, i.e. the $k$-th estimator's underlying random variable lives in its *own* interval, of its own width $b_k - a_k$. Write $\mu^{(k)} := \mathbb{E}[\widehat\mu_m^{(k)}]$.

**Theorem 4.5 (Uniform Hoeffding bound).** For any $\epsilon_1,\ldots,\epsilon_K > 0$,
$$P\Big(\exists k \in [K] : \mu^{(k)} - \widehat\mu_m^{(k)} \geq \epsilon_k\Big) \leq \sum_{k=1}^K \exp\Big(\frac{-2m\epsilon_k^2}{(b_k-a_k)^2}\Big).$$
In the special case $\epsilon_k \equiv \epsilon$ and $b_k - a_k \equiv (b-a)$ for all $k$ (a common range for every estimator), this simplifies to
$$P\Big(\exists k \in [K] : \mu^{(k)} - \widehat\mu_m^{(k)} \geq \epsilon\Big) \leq K \exp\Big(\frac{-2m\epsilon^2}{(b-a)^2}\Big).$$

**Proof.** Let $B_k$ denote the "bad" event $\{\mu^{(k)} - \widehat\mu_m^{(k)} \geq \epsilon_k\}$. Then

$$
\begin{aligned}
P\Big(\exists k \in [K]: \mu^{(k)} - \widehat\mu_m^{(k)} \geq \epsilon_k\Big) &= P\Big(\bigcup_{k=1}^K B_k\Big) \\
&\leq \sum_{k=1}^K P(B_k) \\
&\leq \sum_{k=1}^K \exp\Big(\frac{-2m\epsilon_k^2}{(b_k-a_k)^2}\Big),
\end{aligned}
$$

where the first inequality is the union bound and the second applies Theorem 4.4 (in its lower-tail form) to each $B_k$ separately, using only the independence of $X_1^{(k)}, \ldots, X_m^{(k)}$ *within* estimator $k$ — no relationship between different estimators $k \neq k'$ is required. Note that the $m$ variables of estimator $k$ all lie in the same interval, so $\sum_{i=1}^m (b_k-a_k)^2 = m(b_k-a_k)^2$, which is what turns the $m^2$ in Theorem 4.4 into the $m$ above. $\square$

**What this buys us.** Setting the right-hand side to a target confidence $\delta$ and solving for a common $\epsilon$ (in the equal-range case) gives $\epsilon = (b-a)\sqrt{\log(K/\delta)/(2m)}$, so that simultaneously, for every one of the $K$ estimators,
$$P\Big(\mu^{(k)} \leq \widehat\mu_m^{(k)} + (b-a)\sqrt{\tfrac{\log(K/\delta)}{2m}}, \ \forall k \in [K]\Big) \geq 1-\delta.$$
Two consequences matter directly for what follows in Chapter 5:

* **Effect of the hypothesis class.** If each estimator $k$ corresponds to the training loss of a distinct hypothesis $h_k$ in a finite hypothesis class $\mathcal{H} = \{h_1,\ldots,h_K\}$, the bound degrades only logarithmically in $K = |\mathcal{H}|$ — a much larger hypothesis class costs comparatively little in confidence, but every additional hypothesis we simultaneously monitor must be paid for somewhere, and it is paid for here, inside the square root. This is the seed of the finite-hypothesis-class generalization bound proved in Chapter 5.
* **Effect of loss scale.** If different hypotheses (or different loss functions) produce losses on different scales $[a_k,b_k]$, the *per-estimator* range $(b_k-a_k)$ enters the bound directly and linearly. A hypothesis whose loss can swing across a wide range is intrinsically harder to certify than one confined to a narrow range, even before we ask anything about the hypothesis class's size or structure — boundedness of the *loss*, not just of the sample, is doing real work.
