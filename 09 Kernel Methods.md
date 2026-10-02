# Transforming the input space

Linear predictors have many desirable properties. They:
  - can be trained by analytical methods,
  - are easy to interpret for humans,
  - are theoretically well understood.

However, their expressiveness is limited. For example, they cannot represent the XOR function. The first solution we presented to improve the expressiveness of linear predictors was to pass them through an activation function and build a network of them. The second is **kernel methods**. The core idea is to transform the input space to another space where the problem can be solved by linear methods. For instance, in classification, we expect from the transformed space that there exists a hyperplane that perfectly separates the classes. This way, nonlinearity is handled by the transformation and all the nice properties of linear predictors are preserved. 

Consider a 2D feature space in which a data point $x = (x_1,x_2) \in \mathbb{R}^2$ belongs to one of two classes: one class is concentrated around the origin in the shape of a sphere and the other surrounds this sphere in the shape of a ring. The data are not linearly separable in the original space. However, if we transform the data to a new space $\phi(x) = (x_1^2, x_2^2, \sqrt{2} x_1 x_2) \in \mathbb{R}^3$, the classes become linearly separable as illustrated below.


```python
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['text.usetex'] = True # Enable Latex in plots

# Step 1: Generate random samples for two classes
th.manual_seed(0)

# Number of samples for each class
num_samples = 100

# Class 1:
radius1 = 3
theta1 = 2 * th.pi * th.rand(num_samples)
r1 = th.rand(num_samples).sqrt() * radius1
class1_x = r1 * th.cos(theta1)
class1_y = r1 * th.sin(theta1)

# Class 2
radius2 = 5
theta2 = 2 * th.pi * th.rand(num_samples)
r2 = (-0.5 + th.rand(num_samples)) + radius2
class2_x = r2 * th.cos(theta2)
class2_y = r2 * th.sin(theta2)

# Step 2: Define the transformation function to 3D
def transform_to_3d(x, y):
    x_3d = x**2
    y_3d = y**2
    z_3d = (2 ** 0.5) * x * y
    return x_3d, y_3d, z_3d

# Apply the transformation to both classes
class1_x_3d, class1_y_3d, class1_z_3d = transform_to_3d(class1_x, class1_y)
class2_x_3d, class2_y_3d, class2_z_3d = transform_to_3d(class2_x, class2_y)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# The classes are not linearly separable in 2D
axes[0].scatter(class1_x, class1_y, label='Class 1', c='b')
axes[0].scatter(class2_x, class2_y, label='Class 2', c='r')
axes[0].set_xlabel('$x_1$', fontsize=14)
axes[0].set_ylabel('$x_2$', fontsize=14)
axes[0].set_title('Original 2D Data')
axes[0].legend(loc='best')

# The classes are linearly separable in 3D after transformation
ax = fig.add_subplot(1, 2, 2, projection='3d', proj_type='ortho')

ax.scatter(class1_x_3d, class1_y_3d, class1_z_3d, label='Class 1', c='b')
ax.scatter(class2_x_3d, class2_y_3d, class2_z_3d, label='Class 2', c='r')
ax.set_xlabel('$x_1^2$', fontsize=14)
ax.set_ylabel('$x_2^2$', fontsize=14)
ax.set_zlabel('$\sqrt{2}  x_1  x_2$')

ax.legend(loc='best')
ax.set_title('Transformed Samples in 3D Space')

plt.show()

```


    
![Two classes that are not linearly separable in the original 2D input space (left) become linearly separable after the feature transformation $\phi(x)=(x_1^2,x_2^2,\sqrt{2}x_1x_2)$ (right).](fig/generated/09_Kernel_Methods_1.png)
    


# Kernel Trick

In real-world applications, it is not as straightforward to find a suitable transformation space as in the example above, one that would make the problem linearly separable. The best one can often do is to transform the data to a space that is as high dimensional as possible. However, this is computationally expensive. There are attractive mathematical tools that allow us to model the **similarity** of a pair of data points in the transformation space without explicitly computing the representations of these data points in the transformation space. This can be achieved by a special family of functions, called **kernel functions** with certain characteristics. Given an input space $\mathcal{X}$, we would like to have a kernel function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ with the following property:

$$
   \begin{aligned}
     k(x_i,x_j) =  \phi(x_i)^\top \phi(x_j).
   \end{aligned}
$$

In words, we would like to design directly the left hand side of the equation above without going through the computational steps of the right hand side. This is called the **kernel trick**. This is possible for only certain types of functions that have specific properties. The following result, due to [Mercer (1909)](https://doi.org/10.1098/rsta.1909.0016)<!-- cite: mercer1909functions | article | author={Mercer, James}; title={Functions of Positive and Negative Type, and Their Connection with the Theory of Integral Equations}; journal={Philosophical Transactions of the Royal Society A}; year={1909}; volume={209}; pages={415--446} -->, gives a necessary and sufficient condition for a function to be a valid kernel function.

**Theorem 9.1 (Mercer).** A symmetric function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ admits a feature map $\phi$ with $k(x,x') = \phi(x)^\top\phi(x')$ **if and only if** the
Gram matrix it generates (i.e. the matrix generated by evaluating the function on all pairs of data points in an arbitrary data set with $m$ data points)

$$
\begin{aligned}
    K = \begin{bmatrix}
    k(x_1,x_1) & \cdots & k(x_1,x_m) \\
    \vdots & \ddots & \vdots \\
    k(x_m,x_1) & \cdots & k(x_m,x_m)
    \end{bmatrix}
\end{aligned}
$$

is positive semi-definite, i.e. $v^\top K v \geq 0$ for all $v \in \mathbb{R}^m$.

***Proof of the "only if" direction.*** Suppose $k(x,x') = \phi(x)^\top\phi(x')$ for some feature map $\phi$. Then for any $x_1,\ldots,x_m$ and any $v \in \mathbb{R}^m$,

$$v^\top K v = \sum_{i=1}^m\sum_{j=1}^m v_i v_j\, \phi(x_i)^\top \phi(x_j) = \Big\| \sum_{i=1}^m v_i\, \phi(x_i) \Big\|_2^2 \;\geq\; 0,$$

since the double sum is exactly the squared norm of the single vector $\sum_i v_i \phi(x_i)$. Symmetry of $K$ is immediate from symmetry of the inner product $\square$

The converse — that positive semi-definiteness *suffices*, i.e. that some feature map always exists — is the substantial half of the theorem, and it is proved constructively in the section on reproducing kernel Hilbert spaces below (Theorem 9.2), where the feature map is built as $\phi(x) := k(x,\cdot)$ and the feature space as a space of functions on $\mathcal{X}$. Everything in that construction is carried out here except one analytic step — completing a pre-Hilbert space — which we state but do not perform. Until then, the "only if" direction just proved, together with the composition rules below, is already enough to certify every kernel we actually employ.

The most commonly used kernel functions are:

  - **Linear Kernel:** $k(x,y) =  x^\top y$ 
  - **Polynomial Kernel:** $k(x,y) = \left( x^\top y + c \right)^d$
  - **Radial Basis Function (RBF) Kernel:** $k(x,y) = \exp(-\frac{\| x-y \|^2}{2\sigma^2})$
  - **Sigmoid Kernel:** $k(x,y) = \tanh\left( \gamma  x^\top y  + c \right)$

There also exist kernels that compute the similarity of a pair of graphs, strings, or other complex objects. The nice property of the kernel functions is that one can compute these similarities directly, without needing to define suitable feature spaces for these object types.

One can imagine that it is not straightforward to devise a function that satisfies this condition. However, there are certain rules that allow us to construct new valid kernel functions from existing ones. These rules are called **kernel composition rules**.

**Proposition 9.1 (Kernel composition rules).** Let $k_1, k_2$ be valid kernels, $a \in \mathbb{R}^+$ a constant, and $B$ a positive semi-definite matrix. Then each of the following is a valid kernel:
  - $k(x,x') = k_1(x,x') + k_2(x,x')$
  - $k(x,x') = a \cdot k_1(x,x')$
  - $k(x,x') = k_1(x,x') \cdot k_2(x,x')$
  - $k(x,x') = x^\top B  x'$

***Proof.*** By Theorem 9.1 it suffices to verify that each rule produces a positive semi-definite Gram matrix. Write $K_1, K_2$ for the Gram matrices of $k_1,k_2$ on an arbitrary sample, both positive semi-definite by hypothesis.

*Sum.* $v^\top(K_1+K_2)v = v^\top K_1 v + v^\top K_2 v \geq 0$.

*Positive scaling.* $v^\top(aK_1)v = a\,(v^\top K_1 v) \geq 0$ since $a > 0$.

*Product.* The Gram matrix of the product kernel is the entrywise (Hadamard) product $K_1 \odot K_2$, and the **Schur product theorem** states that the Hadamard product of two positive semi-definite matrices is positive semi-definite. Directly: write $K_2 = \sum_{r} \lambda_r u_r u_r^\top$ with $\lambda_r \geq 0$ (spectral decomposition of a symmetric positive semi-definite matrix). Then for any $v$,
$$v^\top (K_1 \odot K_2) v = \sum_{i,j} v_i v_j (K_1)_{ij} \sum_r \lambda_r u_{r,i}u_{r,j} = \sum_r \lambda_r \underbrace{\big(v \odot u_r\big)^\top K_1 \big(v \odot u_r\big)}_{\geq\,0} \;\geq\; 0 .$$

*Bilinear form.* Since $B$ is symmetric positive semi-definite, it has a square root $B^{1/2}$ with $B = B^{1/2}B^{1/2}$, so $x^\top B x' = (B^{1/2}x)^\top (B^{1/2}x')$ exhibits the feature map $\phi(x) := B^{1/2}x$ directly $\square$

These four rules already certify most kernels used in practice. The polynomial kernel $(x^\top x' + c)^d$, for instance, is built from the linear kernel by $d$ applications of the product rule together with the sum rule and positive scaling (expanding the binomial), and the RBF kernel is obtained as a limit of such combinations.




# The Reproducing Kernel Hilbert Space

Mercer's theorem guarantees that a positive semi-definite kernel *has* a feature map, but says nothing about which one. There are in general many: the polynomial kernel $(x^\top x')^2$ on $\mathbb{R}^2$ is reproduced by $\phi(x) = (x_1^2, x_2^2, \sqrt2 x_1x_2)$ and equally by any rotation of it. One choice is canonical, and it turns out to be the one that lets us stop talking about $\phi$ altogether and talk instead about a **space of functions** — which is what a hypothesis class is. That space is the RKHS of $k$, following the formulation of [Aronszajn (1950)](https://doi.org/10.1090/S0002-9947-1950-0051437-7)<!-- cite: aronszajn1950theory | article | author={Aronszajn, Nachman}; title={Theory of Reproducing Kernels}; journal={Transactions of the American Mathematical Society}; year={1950}; volume={68}; number={3}; pages={337--404} -->.

**Definition 9.1 (Reproducing kernel Hilbert space).** A **reproducing kernel Hilbert space (RKHS)** for a kernel $k$ on $\mathcal{X}$ is a Hilbert space $\mathcal{H}_k$ of functions $f: \mathcal{X}\to\mathbb{R}$, with inner product $\langle\cdot,\cdot\rangle_{\mathcal{H}_k}$ and induced norm $\|f\|_{\mathcal{H}_k} := \sqrt{\langle f,f\rangle_{\mathcal{H}_k}}$, such that

1. $k(x,\cdot) \in \mathcal{H}_k$ for every $x \in \mathcal{X}$, and
2. (**reproducing property**) $\langle f, k(x,\cdot)\rangle_{\mathcal{H}_k} = f(x)$ for every $f \in \mathcal{H}_k$ and every $x \in \mathcal{X}$.

The reproducing property is the whole point: it says that *evaluating* a function at a point is the same operation as taking an inner product against a fixed element of the space. Applying it twice with $f = k(x',\cdot)$ gives

$$\langle k(x,\cdot),\, k(x',\cdot)\rangle_{\mathcal{H}_k} = k(x,x'),$$

so $\phi(x) := k(x,\cdot)$ is a feature map for $k$ — the **canonical feature map** — whose feature space is $\mathcal{H}_k$ itself. Every function in $\mathcal{H}_k$ is simultaneously a hypothesis and a point of the feature space.

---

**Theorem 9.2 (Moore-Aronszajn).** Every positive semi-definite kernel $k$ has exactly one RKHS.

***Proof (construction; completeness omitted).*** Start from the linear span of the kernel sections,

$$\mathcal{V} := \Big\{ f = \sum_{i=1}^n a_i\, k(x_i,\cdot) \;:\; n \in \mathbb{N},\ a_i \in \mathbb{R},\ x_i \in \mathcal{X} \Big\},$$

and define, for $f = \sum_i a_i k(x_i,\cdot)$ and $g = \sum_j b_j k(x_j',\cdot)$,

$$\langle f, g\rangle_{\mathcal{V}} := \sum_{i=1}^n\sum_{j=1}^{n'} a_i b_j\, k(x_i, x_j'). \tag{9.1}$$

*Well defined.* The right-hand side of (9.1) refers to the coefficients $a_i$ and points $x_i$, which are not unique for a given $f$. But rewriting it two ways,

$$\sum_{i,j} a_i b_j k(x_i,x_j') = \sum_{j=1}^{n'} b_j\, f(x_j') = \sum_{i=1}^{n} a_i\, g(x_i),$$

exhibits it once as a quantity depending on $f$ only through its *values*, and once as a quantity depending on $g$ only through its values. Hence it does not depend on either representation.

*Inner product.* Bilinearity is immediate from (9.1) and symmetry from $k(x,x')=k(x',x)$. For positive semi-definiteness, $\langle f,f\rangle_{\mathcal{V}} = \sum_{i,j}a_ia_jk(x_i,x_j) = a^\top K a \geq 0$, which is exactly the Gram-matrix condition of Theorem 9.1 — this is the *only* place the assumption on $k$ is used, and it is used in full.

*Reproducing property.* Taking $g = k(x,\cdot)$ (i.e. $n'=1$, $b_1=1$, $x_1'=x$) in (9.1) gives $\langle f, k(x,\cdot)\rangle_{\mathcal{V}} = \sum_i a_i k(x_i,x) = f(x)$.

*Definiteness.* We must still rule out a non-zero $f$ with $\|f\|_{\mathcal{V}} = 0$. The Cauchy-Schwarz inequality holds for any positive semi-definite bilinear form, so combining it with the reproducing property, for every $x$,

$$|f(x)| = \big|\langle f, k(x,\cdot)\rangle_{\mathcal{V}}\big| \;\leq\; \|f\|_{\mathcal{V}}\,\|k(x,\cdot)\|_{\mathcal{V}} = \|f\|_{\mathcal{V}}\sqrt{k(x,x)}. \tag{9.2}$$

If $\|f\|_{\mathcal{V}}=0$ then $f(x)=0$ for every $x$, i.e. $f$ is the zero *function*. So $\langle\cdot,\cdot\rangle_{\mathcal{V}}$ is a genuine inner product on $\mathcal{V}$, and $(\mathcal{V}, \langle\cdot,\cdot\rangle_{\mathcal{V}})$ is a pre-Hilbert space satisfying both conditions of Definition 9.1.

The remaining step is to **complete** $\mathcal{V}$ — adjoin the limits of its Cauchy sequences — and to check that the completion still consists of functions on $\mathcal{X}$ and still reproduces $k$. Bound (9.2) is what makes this work: convergence in $\|\cdot\|_{\mathcal{V}}$ forces pointwise convergence, so a Cauchy sequence of functions has a well-defined pointwise limit to attach to. Carrying this out, together with the uniqueness claim, is a standard exercise in functional analysis that we do not reproduce here $\square$

Two consequences are worth recording immediately. First, the construction **completes the proof of Mercer's theorem**: the converse direction we deferred in Theorem 9.1 is exactly the statement that a positive semi-definite $k$ admits a feature map, and $\phi(x) := k(x,\cdot)$ into $\mathcal{H}_k$ is one. Second, every element of $\mathcal{H}_k$ is a limit of finite kernel expansions $\sum_i a_i k(x_i,\cdot)$ — so the hypothesis class generated by a kernel is precisely the closure of the set of weighted similarity scores to a finite set of points. This is the formal content of the informal description of kernel methods as "predicting by weighted similarity to remembered examples".

---

**Proposition 9.2 (The RKHS norm controls the function).** For every $f \in \mathcal{H}_k$ and every $x \in \mathcal{X}$, $|f(x)| \leq \|f\|_{\mathcal{H}_k}\sqrt{k(x,x)}$. In particular, if $k(x,x) \leq R^2$ uniformly, then $\sup_x|f(x)| \leq R\|f\|_{\mathcal{H}_k}$.

***Proof.*** This is (9.2), which used only the reproducing property and Cauchy-Schwarz, both available in $\mathcal{H}_k$ $\square$

Proposition 9.2 is the reason $\|\cdot\|_{\mathcal{H}_k}$ is the right complexity measure for kernel methods, and it identifies the hypothesis class of the generalization section below: for a linear predictor $x \mapsto w^\top\phi(x)$ written in the canonical feature map, $w$ *is* a function $f \in \mathcal{H}_k$ and $\|w\|_2$ *is* $\|f\|_{\mathcal{H}_k}$, so

$$\mathcal{H}_B \;=\; \big\{ f \in \mathcal{H}_k : \|f\|_{\mathcal{H}_k} \leq B \big\}$$

is a class of uniformly bounded functions, whatever the dimension of the feature space happens to be. That is precisely the norm-constrained, dimension-free regime that Theorem 5a.4 was built for.

**What the norm measures.** For the RBF kernel the RKHS norm penalizes wiggliness: one can show that $\|f\|^2_{\mathcal{H}_k}$ equals a weighted integral of $|\widehat f(\omega)|^2$ against $e^{\sigma^2\|\omega\|^2/2}$ over the frequency domain, so high-frequency components are charged exponentially, the more so the larger $\sigma$ is. This makes the bandwidth $\sigma$ and the regularization coefficient $\lambda$ two different dials on the same quantity — $\sigma$ decides *which* functions are expensive, $\lambda$ decides *how much* we are willing to pay — and explains, before any experiment, why the fits below vary so sharply with $\sigma$.

---

## The Representer Theorem

The RKHS is typically infinite-dimensional, so minimizing an objective over all of $\mathcal{H}_k$ looks like an infinite-dimensional optimization problem. It is not: the solution is always a finite kernel expansion on the training inputs, with one coefficient per training example. This is the single result that makes kernel methods computable, due to [Kimeldorf & Wahba (1971)](https://doi.org/10.1016/0022-247X(71)90184-3)<!-- cite: kimeldorf1971representer | article | author={Kimeldorf, George S. and Wahba, Grace}; title={Some Results on Tchebycheffian Spline Functions}; journal={Journal of Mathematical Analysis and Applications}; year={1971}; volume={33}; number={1}; pages={82--95} -->.

**Theorem 9.3 (Representer theorem).** Let $S = \{(x_i,y_i)\}_{i=1}^m$, let $\Psi: \mathbb{R}^m \to \mathbb{R}$ be an arbitrary function of the $m$ predicted values (a "data-fit" term), and let $\mathrm{pen}: [0,\infty) \to \mathbb{R}$ be a **strictly increasing** regularizer, in the sense of Chapter 2. Then every minimizer $f^*$ of

$$\min_{f \in \mathcal{H}_k}\ \Psi\big(f(x_1),\ldots,f(x_m)\big) + \mathrm{pen}\big(\|f\|_{\mathcal{H}_k}\big)$$

admits a representation

$$\begin{gathered}
f^*(\cdot) = \sum_{i=1}^m a_i\, k(x_i,\cdot), \\
a \in \mathbb{R}^m.
\end{gathered}$$

***Proof.*** Let $\mathcal{V}_S := \mathrm{span}\{k(x_1,\cdot),\ldots,k(x_m,\cdot)\} \subseteq \mathcal{H}_k$, a finite-dimensional and therefore closed subspace, and decompose any $f \in \mathcal{H}_k$ orthogonally as

$$\begin{gathered}
f = f_\parallel + f_\perp, \\
f_\parallel \in \mathcal{V}_S,\ \ f_\perp \perp \mathcal{V}_S .
\end{gathered}$$

*The data-fit term $\Psi$ cannot see $f_\perp$.* By the reproducing property, for every training input $x_j$,

$$f(x_j) = \langle f, k(x_j,\cdot)\rangle_{\mathcal{H}_k} = \langle f_\parallel, k(x_j,\cdot)\rangle_{\mathcal{H}_k} + \underbrace{\langle f_\perp, k(x_j,\cdot)\rangle_{\mathcal{H}_k}}_{=\,0,\ \text{since } k(x_j,\cdot) \in \mathcal{V}_S} = f_\parallel(x_j),$$

so $\Psi(f(x_1),\ldots,f(x_m)) = \Psi(f_\parallel(x_1),\ldots,f_\parallel(x_m))$ exactly.

*The regularizer can only be hurt by $f_\perp$.* By the Pythagorean identity in a Hilbert space,

$$\|f\|^2_{\mathcal{H}_k} = \|f_\parallel\|^2_{\mathcal{H}_k} + \|f_\perp\|^2_{\mathcal{H}_k} \;\geq\; \|f_\parallel\|^2_{\mathcal{H}_k},$$

with equality if and only if $f_\perp = 0$. Since $\mathrm{pen}$ is strictly increasing, $\mathrm{pen}(\|f\|_{\mathcal{H}_k}) \geq \mathrm{pen}(\|f_\parallel\|_{\mathcal{H}_k})$, strictly so unless $f_\perp = 0$.

Adding the two, the objective at $f_\parallel$ is no larger than at $f$, and is **strictly** smaller whenever $f_\perp \neq 0$. A minimizer therefore has $f^*_\perp = 0$, i.e. $f^* = f^*_\parallel \in \mathcal{V}_S$, which is the claimed form $\square$

Read the proof again for what it does *not* assume. Nothing about $\Psi$: it may be built from the squared loss, the hinge loss of an SVM, the logistic loss, or a non-convex loss with many minima; it need not even be continuous, and it need not decompose across examples at all. The theorem is driven entirely by the reproducing property (which makes $\Psi$ blind to $f_\perp$) and by orthogonality (which makes $\mathrm{pen}$ charge for $f_\perp$ without getting anything in return). What *is* essential is that the regularizer be a strictly increasing function of the RKHS norm specifically — this is why kernel methods regularize with $\|f\|_{\mathcal{H}_k}$ and not with something else.

---

**Corollary 9.1 (Kernel ridge regression).** Take $\Psi(u_1,\ldots,u_m) := \sum_{i=1}^m (y_i - u_i)^2$ and $\mathrm{pen}(t) := \lambda t^2$ with $\lambda > 0$. Then the minimizer over $\mathcal{H}_k$ is $f^*(\cdot) = \sum_i a_i k(x_i,\cdot)$ with

$$a = (K + \lambda I)^{-1} y .$$

***Proof.*** By Theorem 9.3 we may restrict the search to $f = \sum_i a_i k(x_i,\cdot)$, at which point the reproducing property and (9.1) evaluate both terms in terms of the Gram matrix alone:

$$\begin{gathered}
f(x_j) = \sum_i a_i k(x_i,x_j) = (Ka)_j, \\
\|f\|^2_{\mathcal{H}_k} = \sum_{i,j}a_ia_j k(x_i,x_j) = a^\top K a .
\end{gathered}$$

The infinite-dimensional problem has become the $m$-dimensional one

$$\min_{a \in \mathbb{R}^m}\ \|y - Ka\|_2^2 + \lambda\, a^\top K a,$$

whose gradient is $-2Ky + 2KKa + 2\lambda K a = 2K\big[(K+\lambda I)a - y\big]$. Setting it to zero is solved by $(K+\lambda I)a = y$, and $K + \lambda I \succ 0$ for $\lambda>0$ makes that system uniquely solvable $\square$

This is exactly the solution the dual derivation of the previous section arrived at, and exactly what `precompute_a` computes in the implementation below — but obtained without ever mentioning $\phi$, $\Phi$ or $w$. The dual derivation had to *discover* that $w$ lies in the span of the training features, by inspecting the stationarity condition of one particular objective; the representer theorem says it in advance, for every objective of that shape. The practical readings are:

* **Cost.** Training is $O(m^3)$ (one linear solve) and prediction $O(m)$ per query, *independent of the dimension of the feature space* — including when that dimension is infinite, as for the RBF kernel. What is traded away is a cost that grows with the sample size instead, which is the precise sense in which kernel methods are "non-parametric": the model that must be stored is the training set itself.
* **Where the guarantee comes from.** The trained predictor satisfies $\|f^*\|_{\mathcal{H}_k}^2 = a^\top K a \leq \|y\|_2^2/\lambda$ by comparing the objective at $f^*$ against the candidate $f = 0$, so $\lambda$ directly caps the RKHS norm — hence, via Proposition 9.2 and Theorem 5a.4, the Rademacher complexity of the class the solution came from. The section on generalization below makes this quantitative.

# Kernel regression

Let us remember the linear methods. It is possible to apply the kernel trick to all of them, i.e. to **kernelize** them. Let us take **ridge regression** as an example. Consider the case where the raw inputs $x_i$ are passed through a feature extraction step $\phi(x_i)$. Call the matrix that contains $\phi(x_i)^\top$ in its rows $\Phi \in \mathbb{R}^{m \times d}$. The ridge regression objective can then be expressed as

$$
\begin{aligned}
    \mathcal{L}(w) &= \sum_{i=1}^m (y_i - w^\top \phi(x_i))^2 + \lambda ||w||_2^2\\
      &= y^\top y - 2y^\top \Phi w + w^\top \Phi^\top \Phi w + \lambda w^\top w,
\end{aligned}
$$

where $y \in \mathbb{R}^m$ is the target vector and $w \in \mathbb{R}^d$ is the weight vector. We perform learning by minimizing this objective with respect to the weights:

$$
\begin{aligned}
  \nabla_{w} \mathcal{L}(w) &= -2\Phi^\top y + 2\Phi^\top \Phi w + 2\lambda w \triangleq 0 \\
    &\Rightarrow (\Phi^\top \Phi + \lambda I) w = \Phi^\top y \\
    &\Rightarrow w = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top y.
\end{aligned}
$$

This primal solution still requires inverting a $d \times d$ matrix and computing $\phi(\cdot)$ explicitly, both of which can be prohibitive when $\phi$ maps into a very high (or infinite) dimensional space. Since the gradient stationarity condition $\Phi^\top \Phi w + \lambda w = \Phi^\top y$ implies $w = \frac{1}{\lambda}\Phi^\top(y - \Phi w)$, the optimal $w$ always lies in the span of the training feature vectors $\phi(x_1), \ldots, \phi(x_m)$. We can therefore reparametrize it as

$$
\begin{aligned}
  w = \Phi^\top a, \qquad a \in \mathbb{R}^m,
\end{aligned}
$$

and search for the optimal $a$ instead. Placing this expression into the objective gives its so-called **dual representation**, which contains the data only through inner products in the transformed space:

$$
\begin{aligned}
      \mathcal{L}(a) &= y^\top y - 2y^\top \Phi \Phi^\top a + a^\top \Phi \Phi^\top \Phi \Phi^\top a + \lambda a^\top \Phi \Phi^\top a.
\end{aligned}
$$

Note that $K := \Phi \Phi^\top$ is the $m \times m$ dimensional **Gram** matrix with $K_{ij} = \phi(x_i)^\top \phi(x_j)$. This inner product can be replaced by any kernel function, generating a Gram matrix with entries $K_{ij} = k(x_i,x_j)$. The kernelized objective then reads

$$
\begin{aligned}
 \mathcal{L}(a) = y^\top y - 2y^\top K a + a^\top K K a + \lambda a^\top K a.
\end{aligned}
$$

Setting the gradient of $\mathcal{L}$ with respect to $a$ to zero and solving for $a$ gives

$$
 \begin{aligned}
 \nabla_{a} \mathcal{L}(a) &= -2 K y + 2 K K a + 2 \lambda K a \triangleq 0 \\
 \Rightarrow & K \big[ (K + \lambda I) a - y \big] = 0 \\
 \Rightarrow & a = (K+\lambda I)^{-1} y,
\end{aligned}
$$

where the last step assumes $K$ is invertible (it always is a valid solution regardless, since any $a$ satisfying $(K+\lambda I)a = y$ also solves $K\big[(K+\lambda I)a - y\big]=0$).

To reiterate, we first redefined the parameters $w$ by an expression that involves a newly introduced variable $a$, and then found the optimal value for this variable that minimizes the training loss. The resulting objective does not have any trainable parameters left that are not summarized by the Gram matrix $K$. Furthermore, the optimal value of $a$ requires generating a Gram matrix using the whole training set, and it appears as an intermediate result in the calculation of the prediction function. We predict the label $y_*$ of a query input $x_*$ by

$$
\begin{aligned}
    f(x_*) &= \phi(x_*)^\top w = \phi(x_*)^\top \Phi^\top a \\
        &= k_*^\top (K+\lambda I)^{-1} y,
\end{aligned}
$$

where

$$
\begin{aligned}
    k_* = \begin{bmatrix}
             k(x_*, x_1)\\
             k(x_*, x_2)\\
             \vdots,\\
             k(x_*, x_m)
          \end{bmatrix}.   
\end{aligned}
$$

Because the prediction function requires computations on the whole training set, kernel regression is called a **lazy learner**: unlike the models we have seen so far, it does all of its work at prediction time, not at training time (the k-nearest neighbor approach of Chapter 3 is the other example — "training" there is nothing more than memorizing the data set). It is also a **non-parametric** method. As seen in the example implementation given below, its "predict" function takes the whole training set as an input in order to calculate the $k_*$ vector.

A major weakness of the kernel methods is that many common kernel functions have tunable hyperparameters. For instance, the RBF kernel has a hyperparameter $\sigma$ that controls the length scale of the kernel — and, as the RKHS reading above showed, thereby decides how expensive high-frequency components are in $\|f\|_{\mathcal{H}_k}$. The performance of the model is sensitive to the choice of this hyperparameter. The example below performs kernel regression with three different choices of $\sigma$, each giving dramatically different outcomes. While $\sigma=0.1$ gives a reasonable fit, $\sigma=0.01$ **overfits** — the kernel sections are so narrow that the fit spikes at each training point and returns to zero between them — and $\sigma=1$ **underfits**, since at that length scale the RKHS charges so much for curvature that the fit cannot follow a full period of the sine.


```python
import matplotlib.pyplot as plt
import torch as th

th.manual_seed(6)

# Generate the data
inputs = th.linspace(0, 1, 50, dtype=th.float64).unsqueeze(1)
outputs = th.sin(2 * th.pi * inputs)
labels = outputs + th.randn(inputs.shape, dtype=th.float64) * 0.25

num_samples = inputs.shape[0]
num_train_samples = num_samples // 2

idx = th.randperm(num_samples)
inputs_train = inputs[idx[:num_train_samples]]
labels_train = labels[idx[:num_train_samples]]

class KernelLinearRegression:
    def __init__(self, lambda_reg=0.1, length_scale=0.1):
        self.lambda_reg = lambda_reg
        self.sigma = length_scale

    def compute_rbf_kernel(self, X, Xp):
        # k(x,x') = exp(-||x-x'||^2 / (2 sigma^2))
        dist = th.cdist(X, Xp)
        return th.exp(-dist**2 / (2 * self.sigma**2))

    def precompute_a(self, inputs_train, labels_train):
        # a = (K + lambda I)^{-1} y, solved as the linear system
        # (K + lambda I) a = y
        K = self.compute_rbf_kernel(inputs_train, inputs_train)
        A = K + self.lambda_reg * th.eye(inputs_train.shape[0], dtype=K.dtype)
        self.a = th.linalg.solve(A, labels_train)

    def predict(self, inputs_train, inputs):
        # h(x) = sum_i a_i k(x_i, x)
        Kp = self.compute_rbf_kernel(inputs_train, inputs)
        return Kp.T @ self.a

# Create three models with different kernel hyperparameters
model1 = KernelLinearRegression(length_scale=0.01)
model1.precompute_a(inputs_train, labels_train)

model2 = KernelLinearRegression(length_scale=0.1)
model2.precompute_a(inputs_train, labels_train)

model3 = KernelLinearRegression(length_scale=1)
model3.precompute_a(inputs_train, labels_train)

# This is a "nonparametric model", so
#   i)  There is no "learn" function. "precompute_a" only
#       computes a value that is reused across different predictions
#   ii) Predict function takes the training data as input

predictions1 = model1.predict(inputs_train, inputs)
predictions2 = model2.predict(inputs_train, inputs)
predictions3 = model3.predict(inputs_train, inputs)

plt.plot(inputs, outputs, 'k-',
         label="True function", linewidth=4)
plt.plot(inputs_train, labels_train, 'bo',
         label="Training data")
plt.plot(inputs, predictions1, 'r--',
         label=r"Predictions ($\sigma=0.01$)", linewidth=3)
plt.plot(inputs, predictions2, 'g--',
         label=r"Predictions ($\sigma=0.1$)", linewidth=3)
plt.plot(inputs, predictions3, 'm--',
         label=r"Predictions ($\sigma=1$)", linewidth=3)
plt.legend(loc="upper right")
plt.show()
```


    
![Kernel ridge regression fits with an RBF kernel of bandwidth $\sigma \in \{0.01, 0.1, 1\}$, illustrating overfitting, a good fit, and underfitting.](fig/generated/09_Kernel_Methods_2.png)
    


## Generalization of Kernel Methods

Kernel methods look, at first glance, like exactly the setting Theorem 5a.4 warned us about: the feature map $\phi(x)$ can live in an arbitrarily high — even infinite — dimensional space (the RBF kernel above is a standard example), for which $d_{VC}$ is typically infinite, making the VC-dimension bounds of Chapter 5 vacuous. But Theorem 5a.4 already told us this does not matter: for a norm-constrained linear predictor $x \mapsto w^\top \phi(x)$ with $\|w\|_2 \leq B$, the Rademacher complexity is $BR/\sqrt m$, where $R$ bounds $\|\phi(x)\|_2$ — with **no dependence on the dimension of $\phi(x)$ at all**. This is precisely the regime kernel methods create by design, and in the canonical feature map of Definition 9.1 the constrained class is exactly the RKHS ball $\mathcal{H}_B = \{f \in \mathcal{H}_k : \|f\|_{\mathcal{H}_k} \leq B\}$ of Proposition 9.2.

We can, in fact, do slightly better and eliminate $\phi$ from the bound entirely, expressing it purely in terms of the kernel — a small derivation worth doing explicitly, since it shows the kernel trick paying off even inside the generalization theory itself, not only inside the training algorithm. Let $\mathcal{H}_B := \{x \mapsto w^\top \phi(x) : \|w\|_2 \leq B\}$. Since $\|\phi(x)\|_2^2 = \phi(x)^\top\phi(x) = k(x,x)$,

$$
\begin{aligned}
\widehat{\mathfrak{R}}_S(\mathcal{H}_B) = \frac{B}{m}\, \mathbb{E}_\sigma\Big[\Big\|\sum_{i=1}^m \sigma_i \phi(x_i)\Big\|_2\Big] &\leq \frac{B}{m}\sqrt{\mathbb{E}_\sigma\Big[\Big\|\sum_i \sigma_i \phi(x_i)\Big\|_2^2\Big]} \\
&= \frac{B}{m}\sqrt{\sum_{i=1}^m \|\phi(x_i)\|_2^2} = \frac{B}{m}\sqrt{\sum_{i=1}^m k(x_i,x_i)} = \frac{B\sqrt{\mathrm{tr}(K)}}{m},
\end{aligned}
$$

using Jensen's inequality and $\mathbb{E}[\sigma_i\sigma_{i'}]=0$ for $i \neq i'$ (exactly as in the proof of Theorem 5a.4), where $K$ is the Gram matrix of Mercer's theorem. This **trace bound** is computable directly from the kernel matrix — never requiring $\phi$ to be written down — and is always at least as tight as the generic $BR/\sqrt m$ bound, since $\mathrm{tr}(K) = \sum_i k(x_i,x_i) \leq mR^2$ whenever $k(x,x)\leq R^2$ uniformly.

This connects directly to the regularization coefficient $\lambda$ already introduced for ridge regression in Chapter 2 and reused for kernel regression above. At the optimum $w^*$ of the ridge objective $\mathcal{L}(w) = \|y-\Phi w\|_2^2 + \lambda\|w\|_2^2$, optimality against the trivial candidate $w=0$ gives $\lambda\|w^*\|_2^2 \leq \mathcal{L}(w^*) \leq \mathcal{L}(0) = \|y\|_2^2$, hence

$$\|w^*\|_2 \leq \frac{\|y\|_2}{\sqrt{\lambda}} =: B(\lambda).$$

Plugging $B(\lambda)$ into the trace bound above makes quantitative exactly what Chapter 2 promised only qualitatively when it introduced ridge regression as an instance of Structural Risk Minimization: increasing $\lambda$ shrinks $B(\lambda)$, which shrinks $\widehat{\mathfrak{R}}_S(\mathcal{H}_{B(\lambda)})$, which — through Theorem 5a.2 — tightens the generalization guarantee. The regularization coefficient is not merely a heuristic device against overfitting; it is a direct dial on the Rademacher complexity of the space the ERM solution is allowed to come from.

## The Neural Tangent Kernel

Chapter 8 analyzed a two-layer network $h_{W,w}(x) = w^\top g(Wx)$ and bounded its Rademacher complexity by controlling the norms $B_1, B_2$ of *both* layers — a bound that holds regardless of how $W$ and $w$ are jointly trained. There is an illustrative special case of that same network in which training reduces *exactly* to the kernel methods of this chapter, and it is worth making explicit, since it is the seed of one of the more influential ideas in modern deep learning theory: the **Neural Tangent Kernel (NTK)**.

**A simplified case: random features.** Suppose we freeze the hidden layer at its random initialization $W_0$ (drawn once, e.g. with i.i.d. $\mathcal{N}(0, 1/d)$ entries per row) and train *only* the output layer $w$. Then $\phi(x) := g(W_0 x)$ is a fixed feature map, and $h_{W_0,w}(x) = w^\top \phi(x)$ is exactly a kernel method with kernel $k_{W_0}(x,x') := \phi(x)^\top\phi(x') = \sum_{j=1}^k g(w_{0,j}^{(1)\top}x)\,g(w_{0,j}^{(1)\top}x')$ — every tool from this chapter, including the generalization bound just derived, applies to it unchanged. As the width $k \to \infty$, the weak law of large numbers (Theorem 4.3, applied to the $k$ i.i.d. terms $g(w_{0,j}^{(1)\top}x)\,g(w_{0,j}^{(1)\top}x')$ as $j$ ranges over hidden units) gives

$$\frac{1}{k}\, k_{W_0}(x,x') \;\xrightarrow{\ P\ }\; k_\infty(x,x') := \mathbb{E}_{u \sim \mathcal{N}(0,I/d)}\big[g(u^\top x)\,g(u^\top x')\big],$$

a **deterministic kernel that no longer depends on the random draw of $W_0$** at all, only on the architecture ($g$) and the input distribution of $u$. This is the "random features" or "infinite-width random network" kernel, and it already illustrates the core NTK phenomenon in the easiest possible case: a wide random network with a trained linear readout *is*, in the limit, exactly kernel regression with a fixed, explicitly computable kernel.

**The general case.** The full Neural Tangent Kernel result — which trains *all* parameters $\theta = (W,w)$, not just the output layer — is more subtle but built on the same idea. Near the initialization $\theta_0$, a first-order Taylor expansion of the network output in its parameters gives

$$f_\theta(x) \approx f_{\theta_0}(x) + \nabla_\theta f_{\theta_0}(x)^\top (\theta - \theta_0),$$

which is *affine* in $\theta$: exactly the linear-predictor template of this chapter once more, but now with the **tangent feature map** $\phi_{\mathrm{NTK}}(x) := \nabla_\theta f_{\theta_0}(x)$ (one feature per network parameter) and its associated kernel $k_{\mathrm{NTK}}(x,x') := \nabla_\theta f_{\theta_0}(x)^\top \nabla_\theta f_{\theta_0}(x')$. Jacot, Gabriel & Hongler (2018) show two facts that make this linearization exact rather than merely suggestive, in the infinite-width limit and for appropriately scaled initializations: (i) $k_{\mathrm{NTK}}$ converges, by the same law-of-large-numbers mechanism as above (now averaging over a growing number of parameters instead of a growing number of hidden units), to a fixed deterministic kernel depending only on the architecture; and (ii) gradient descent moves $\theta$ so little from $\theta_0$ relative to the network's width that the linearization above remains accurate for the *entire* training trajectory, not just near initialization. Under these conditions, training an infinitely wide neural network by gradient descent is equivalent to kernel regression (as derived earlier in this chapter) against the fixed kernel $k_{\mathrm{NTK}}$.

**Why this matters for generalization.** This equivalence resolves, for the infinite-width regime, the very puzzle Chapter 8's PAC analysis raised: an infinitely wide network has infinite VC dimension, yet the NTK correspondence shows it is *simultaneously* a kernel predictor of the exact form Theorem 5a.4 and the trace bound above already know how to handle — capacity controlled by $\|w\|_{\mathcal H}$ (the RKHS norm under $k_{\mathrm{NTK}}$) and by $\mathrm{tr}(K_{\mathrm{NTK}})$, neither of which explodes with the width. This is the precise sense in which Chapter 8's remark that "capacity control by norm, rather than by counting parameters, is what makes the bound meaningful" for neural networks is not a coincidence but a theorem: in the wide-network limit, a neural network trained by gradient descent literally *is* a norm-constrained kernel predictor, and the same Rademacher-complexity machinery developed once, in Chapter 5a, governs both.

# Support Vector Machines

A prime application of kernel methods, and the historical reason the representer theorem is stated in the generality it is, is the **Support Vector Machine (SVM)** [Boser et al. (1992)](https://doi.org/10.1145/130385.130401)<!-- cite: boser1992training | inproceedings | author={Boser, Bernhard E. and Guyon, Isabelle M. and Vapnik, Vladimir N.}; title={A Training Algorithm for Optimal Margin Classifiers}; booktitle={Proceedings of the 5th Annual Workshop on Computational Learning Theory (COLT)}; year={1992}; pages={144--152} -->. Among many possible hyperplanes that may separate the data points of two classes, SVM chooses the one that maximizes the smallest distance between a data point and the hyperplane. This distance is called the **margin**. Because of this property, SVM is also called a **maximum margin classifier**. The data points that are closest to the hyperplane are called **support vectors**. The margin-maximizing hyperplane is defined by only these support vectors. In the language of Theorem 9.3, the SVM objective is an instance of the representer template with the **hinge** data-fit term $\Psi(u) = \sum_i \max(0, 1-y_iu_i)$ and $\mathrm{pen}(t) = \lambda t^2$, so its solution is again $f^* = \sum_i a_i k(x_i,\cdot)$ — but the hinge loss, unlike the squared loss, is flat on correctly classified points beyond the margin, which forces $a_i = 0$ for all of them. The support vectors are exactly the indices with $a_i \neq 0$. The other data points do not play any role in prediction. This is a desirable property especially when the SVM is kernelized, as the kernel function is evaluated only on the support vectors instead of the whole training set as was the case for kernel regression above.  The figure below illustrates the core idea behind the SVM. We leave the mathematical details out of our scope, since SVMs have been greatly overshadowed by deep learning and Gaussian processes in modern machine learning practice.


```{=latex}
\input{fig/svm.tex}
```


