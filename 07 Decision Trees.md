Every hypothesis class we have used so far predicts through a fixed algebraic form: a dot product for linear predictors (Chapter 3), a sigmoid or softmax of a dot product for classifiers (Chapter 4), a weighted sum of similarities for kernel methods (Chapter 11). A **decision tree** [Breiman et al. (1984)](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)<!-- cite: breiman1984cart | book | author={Breiman, Leo and Friedman, Jerome H. and Olshen, Richard A. and Stone, Charles J.}; title={Classification and Regression Trees}; publisher={Wadsworth}; year={1984} --> predicts through a different mechanism entirely: a sequence of "if-then" questions about individual features, organized in a tree, that recursively partitions the feature space $\mathcal{X}$ into regions, each assigned a single output.

# The Hypothesis Class

A decision tree is a rooted tree in which:

* every **internal node** $v$ is labeled with a **splitting rule** $\phi_v: \mathcal{X} \rightarrow \{1,\ldots,k_v\}$ that routes an input $x$ to one of $v$'s $k_v$ children. We focus on binary trees, $k_v=2$, and on two rule types: for a real-valued feature, $\phi_v(x) := \mathds{1}\big(x_{j(v)} \leq t_v\big)$ for a feature index $j(v) \in [d]$ and a node-specific threshold $t_v \in \mathbb{R}$; for a categorical feature, $\phi_v(x) := \mathds{1}\big(x_{j(v)} = a_v\big)$ for a node-specific category $a_v$. The index $j(v)$, the threshold $t_v$, and the category $a_v$ are node-specific: each internal node chooses its own test, and two nodes may test the same feature at different thresholds,
* every **leaf** $u$ is labeled with a prediction $c_u \in \mathcal{Y}$.

A tree $T$ computes a hypothesis $h_T: \mathcal{X} \rightarrow \mathcal{Y}$ by routing $x$ from the root, following $\phi_v(x)$ at each internal node $v$, until it reaches a leaf $u$, and outputting $h_T(x) := c_u$. Since each leaf's region is exactly the intersection of the splitting rules on the path from the root, the collection of leaves of $T$ partitions $\mathcal{X}$ into disjoint axis-aligned regions, and $h_T$ is piecewise constant on this partition. The **hypothesis class of decision trees**, $\mathcal{H}_{\text{tree}}$, is the union of $h_T$ over all finite binary trees $T$ built from the allowed splitting rules — an infinite, nonuniform hypothesis class in the sense of Chapter 6, since larger trees induce finer partitions and hence more expressive hypotheses.

Formally, a decision tree instantiates the general recursive partitioning scheme

$$h_T(x) = \sum_{u \in \mathrm{leaves}(T)} c_u \cdot \mathds{1}\big(x \in R_u\big),$$

where the region of leaf $u$ is obtained by intersecting the half-spaces selected along its path. Let $v_L$ and $v_R$ denote the left and right child of an internal node $v$, and let $\mathrm{leaves}(v)$ denote the leaves of the subtree rooted at $v$. For threshold splits,

$$R_u := \bigcap_{v \in \mathrm{path}(u)} \begin{cases} \{x \in \mathcal{X} : x_{j(v)} \leq t_v\}, & u \in \mathrm{leaves}(v_L), \\ \{x \in \mathcal{X} : x_{j(v)} > t_v\}, & u \in \mathrm{leaves}(v_R), \end{cases}$$

and for categorical splits the same display holds with $=$ and $\neq$ replacing $\leq$ and $>$. The family $\{R_u\}$ over $u \in \mathrm{leaves}(T)$ partitions $\mathcal{X}$.

Figure 9.1 displays the two objects side by side: the recursive partition of a two-dimensional feature space and the tree that generates it. Figure 9.2 displays the graph of the resulting predictor, a piecewise-constant surface whose plateaus are the leaf predictions. The construction is the standard recursive-partitioning representation: every internal node splits one region along a feature axis, and every leaf owns one cell of the resulting partition.

```{=latex}
\input{fig/decision_tree_partition.tex}
```

```{=latex}
\input{fig/decision_tree_surface.tex}
```

# Growing a Tree: A Greedy ERM Surrogate

Given a training set $S$, ERM over $\mathcal{H}_{\text{tree}}$ searches all trees for the one minimizing $\widehat{R}_S(h_T)$. The search cannot be carried out efficiently in general: the following proposition shows that even the feasibility question — whether some tree with at most $L$ leaves fits the sample exactly — is NP-complete. The intractability therefore lies in the tree structure and the choice of tests jointly, not in a particular impurity criterion.

**Proposition 9.2.1 (Leaf-budgeted ERM for decision trees is NP-hard).** Let $\mathcal{X}=\{0,1\}^d$, $\mathcal{Y}=\{0,1\}$, and let $\mathcal{H}_{\mathrm{tree},L}^{\mathrm{cat}}$ be the class of binary decision trees with at most $L$ leaves whose tests are of the form $\mathds{1}(x_j=1)$. Given a sample $S$ and an integer $L$, deciding whether some $h\in\mathcal{H}_{\mathrm{tree},L}^{\mathrm{cat}}$ satisfies $\widehat{R}_S(h)=0$ is NP-complete.

***Proof.*** Membership in NP is immediate: a certificate is a tree with at most $L$ leaves, its tests, and its leaf labels, and consistency with $S$ is verified by routing each of the $m$ examples. The certificate can be taken of polynomial size in $d$ and $m$: if a consistent tree exists, contracting nodes with a single child and pruning leaves that contain no sample point yields a tree whose leaves each contain at least one of the $m$ examples, hence a tree with at most $m$ leaves, and its tests involve at most $d$ features.

For NP-hardness, reduce from Hitting Set: given a universe $[d]$, a family $A_1,\ldots,A_n\subseteq[d]$, and an integer $k$, decide whether some $H\subseteq[d]$ with $|H|\le k$ intersects every $A_i$. Construct $S$ from the $n$ positive examples $x_i:=\mathds{1}_{A_i}$, $y_i:=1$, and one negative example $x_0:=0^d$, $y_0:=0$. We claim that a consistent tree with at most $k+1$ leaves exists if and only if a hitting set of size at most $k$ exists.

($\Leftarrow$) Given $H=\{j_1,\ldots,j_k\}$, build a chain that tests $x_{j_1}=1,\ldots,x_{j_k}=1$ in order: at the $t$-th test, the branch $x_{j_t}=1$ ends in a leaf labeled $1$, and the branch $x_{j_t}=0$ continues to the next test; the final $x_{j_k}=0$ branch ends in a leaf labeled $0$. The tree has $k+1$ leaves. Every positive $x_i$ has $x_{ij}=1$ for some $j\in H$, so it exits at a positive leaf; $x_0$ takes every zero branch and reaches the negative leaf.

($\Rightarrow$) Let $T$ be a consistent tree with at most $k+1$ leaves, and follow the path of $x_0$ from the root. At each internal node on this path, $x_0$ takes the branch $x_j=0$; let $H$ be the set of features tested on the path. The path contains at most $k$ internal nodes, since a binary tree with $\ell$ leaves has $\ell-1$ internal nodes, so $|H|\le k$. For any positive $x_i$, if it took the zero branch at every node of the path, it would follow the same route as $x_0$ and end in the leaf labeled $0$, contradicting consistency. Hence at some node on the path $x_i$ takes the branch $x_j=1$, which requires $x_{ij}=1$, so $j\in A_i$. Thus $H$ intersects every $A_i$, and $|H|\le k$. $\square$

The reduction uses only equality tests on binary features; threshold tests $\mathds{1}(x_j\le t)$ with $0\le t<1$ induce the same partition of $\{0,1\}^d$, so the hardness carries over to the threshold model used in this chapter. Hitting Set is one of the original NP-complete problems [Karp (1972)](https://doi.org/10.1007/978-1-4684-2001-2_9)<!-- cite: karp1972reducibility | inproceedings | author={Karp, Richard M.}; title={Reducibility Among Combinatorial Problems}; booktitle={Complexity of Computer Computations}; year={1972} -->, and the NP-completeness of constructing optimal binary decision trees is the classical result [Hyafil & Rivest (1976)](https://doi.org/10.1016/0020-0190(76)90095-8)<!-- cite: hyafil1976constructing | article | author={Hyafil, Laurent and Rivest, Ronald L.}; title={Constructing Optimal Binary Decision Trees Is NP-Complete}; journal={Information Processing Letters}; year={1976}; volume={5}; number={1}; pages={15--17} -->. The proposition is the leaf-budgeted consistency version of that result: it isolates the combinatorial core of the problem and shows that the intractability is present already in the feasibility question, before any impurity criterion is chosen.

In practice, we build the tree by a **greedy, recursive** procedure:

1. Start with a single leaf holding all of $S$.
2. At each leaf currently holding a subset $S_v \subseteq S$, consider splitting it into two children by some rule $\phi_v$. Among all candidate rules (all features, all thresholds), pick the one that most reduces an **impurity** measure of the label distribution within the node (defined below).
3. Recurse into each child with its corresponding subset of $S_v$, and repeat, until a stopping criterion is met (e.g. a maximum depth, a minimum leaf size, or zero remaining impurity).
4. Label each final leaf $u$ with the majority class of $S_v$ restricted to that leaf (classification) or the mean label (regression).

The key design choice is the **impurity measure**, which stands in for $\widehat{R}_S(h)$ as a differentiable-in-spirit, locally computable proxy that the greedy procedure can optimize one split at a time. For a node $v$ holding a subset $S_v$, with class proportions $\widehat{p}_c := \frac{|\{(x,y) \in S_v : y=c\}|}{|S_v|}$ for $c \in \{1,\ldots,C\}$, three standard choices are:

* **Entropy** (of the empirical label distribution at the node): $H(S_v) := -\sum_{c=1}^C \widehat{p}_c \log \widehat{p}_c$, borrowed directly from information theory — it is maximized when $S_v$'s labels are uniformly spread across classes (there is the most "uncertainty" about the label at this node) and zero when $S_v$ is pure (a single class). Following the convention of these notes, $\log$ here is the natural logarithm, so entropy is measured in nats rather than the more familiar bits; the base only rescales $H$ by a positive constant and therefore never changes which split is selected.
* **Gini index**: $G(S_v) := 1 - \sum_{c=1}^C \widehat{p}_c^2 = \sum_{c=1}^C \widehat{p}_c(1-\widehat p_c)$, the probability that two labels drawn independently (with replacement) from $S_v$'s empirical distribution disagree. It behaves similarly to entropy — zero for a pure node, maximal for a uniform one — but is less costly to compute and does not involve a logarithm.
* **Misclassification error**: $E(S_v) := 1 - \max_{c \in [C]} \widehat{p}_c$, the fraction of examples that do not belong to the majority class. It is the empirical $0/1$ risk of the majority-vote leaf and serves as the natural benchmark, but as a split criterion it is coarse: for two classes it is piecewise linear in the class proportion, so it treats as equivalent many splits that Gini and entropy rank differently.

```{=latex}
\input{fig/impurity_measures.tex}
```

Figure 9.3 compares the three measures for two classes, as functions of the proportion $p$ of the second class. All three vanish at the pure nodes $p=0$ and $p=1$, and all three attain their maximum at $p=1/2$. Misclassification error is piecewise linear with a kink at $p=1/2$, while the Gini index and entropy are strictly concave and respond more sharply away from the balance point. The difference matters for split selection. With $1000$ examples, $500$ per class, suppose one split produces children with class counts $(375,125)$ and $(125,375)$, while another produces $(250,500)$ and $(250,0)$. Both splits have weighted misclassification error $1/4$: in the first, each child has minority fraction $1/4$; in the second, the impure child has minority fraction $1/3$ but carries only $3/4$ of the mass. The second split has Gini index $1/3$ against $3/8$ for the first, and lower entropy as well, because it isolates a pure node. Tree growth therefore uses Gini or entropy, while misclassification error is appropriate for pruning, where the number of leaves is already controlled and the quantity being optimized is the empirical error of the pruned tree.

For growth we use the two differentiable measures, so let $\mathrm{Imp}(\cdot) \in \{H, G\}$ in the following definition. Given a candidate split of $S_v$ into $S_{v,\text{left}}$ and $S_{v,\text{right}}$, the **information gain** of the split is the reduction in impurity, weighted by the resulting subset sizes:

$$\mathrm{Gain}(S_v, \phi_v) := \mathrm{Imp}(S_v) - \frac{|S_{v,\text{left}}|}{|S_v|}\mathrm{Imp}(S_{v,\text{left}}) - \frac{|S_{v,\text{right}}|}{|S_v|}\mathrm{Imp}(S_{v,\text{right}}).$$

At each node, the greedy algorithm picks the split $\phi_v$ maximizing $\mathrm{Gain}(S_v,\phi_v)$ over all candidate features and thresholds. This greedy, node-by-node maximization is what makes decision tree induction tractable: it never revisits an earlier split, and never jointly optimizes two splits together, with the consequence that it is no longer guaranteed to find the ERM-optimal tree of a given size.

# Illustrative Example

Consider a tiny data set for deciding whether to play outdoor tennis, with two binary features — $\mathtt{Outlook}$ taking the values Sunny and Rainy, and $\mathtt{Wind}$ taking the values Weak and Strong — and label $\mathtt{Play}$ taking the values Yes and No:

| $\mathtt{Outlook}$ | $\mathtt{Wind}$ | $\mathtt{Play}$ |
|---|---|---|
| Sunny | Weak | Yes |
| Sunny | Weak | Yes |
| Sunny | Weak | Yes |
| Sunny | Strong | No |
| Rainy | Weak | No |
| Rainy | Weak | No |
| Rainy | Strong | No |
| Rainy | Strong | No |

Here $m=8$, with $3$ "Yes" and $5$ "No", so the root's Gini index is
$$G(S) = 1 - \Big(\tfrac38\Big)^2 - \Big(\tfrac58\Big)^2 = \tfrac{15}{32} = 0.46875.$$
The counts are read directly from the table: the first three rows are Yes, and the remaining five are No.

**Candidate split on $\mathtt{Outlook}$.** This sends the $4$ Sunny points (3 Yes, 1 No) to the left child and the $4$ Rainy points (0 Yes, 4 No) to the right child:

$$\begin{gathered}
G(S_{\text{left}}) = 1 - \big(\tfrac34\big)^2 - \big(\tfrac14\big)^2 = 0.375, \\
G(S_{\text{right}}) = 1 - 1^2 - 0^2 = 0.
\end{gathered}$$

$$\mathrm{Gain}(S, \mathtt{Outlook}) = \tfrac{15}{32} - \tfrac{4}{8}\big(\tfrac{6}{16}\big) - \tfrac{4}{8}(0) = \tfrac{15}{32} - \tfrac{6}{32} = \tfrac{9}{32} = 0.28125.$$

**Candidate split on $\mathtt{Wind}$.** This sends the $5$ Weak points (3 Yes, 2 No) to the left child and the $3$ Strong points (0 Yes, 3 No) to the right child:

$$\begin{gathered}
G(S_{\text{left}}) = 1 - \big(\tfrac35\big)^2 - \big(\tfrac25\big)^2 = 0.48, \\
G(S_{\text{right}}) = 1 - 0^2 - 1^2 = 0.
\end{gathered}$$

$$\mathrm{Gain}(S, \mathtt{Wind}) = \tfrac{15}{32} - \tfrac{5}{8}\big(\tfrac{12}{25}\big) - \tfrac{3}{8}(0) = \tfrac{15}{32} - \tfrac{3}{10} = \tfrac{27}{160} = 0.16875.$$

Since $\mathrm{Gain}(S,\mathtt{Outlook}) = \tfrac{9}{32} = 0.28125 > \tfrac{27}{160} = 0.16875 = \mathrm{Gain}(S,\mathtt{Wind})$, the greedy algorithm splits on $\mathtt{Outlook}$ first. The right child (Rainy) is already pure ($G=0$) and becomes a leaf predicting $\mathrm{No}$. The left child (Sunny, $3$ Yes/$1$ No) is not pure, so we recurse: splitting it further on $\mathtt{Wind}$ sends its single Strong/No point to its own leaf and the remaining $3$ Weak/Yes points to another, both pure, terminating the recursion. The resulting tree is

```
                 Outlook?
               /          \
          Sunny             Rainy
            |                 |
          Wind?              No
        /        \
     Weak        Strong
      |            |
     Yes           No
```

Observe that this tree achieves $\widehat{R}_S(h_T) = 0$ with three leaves. Three is also the minimum for this sample: a single split cannot separate the classes, since both candidate roots leave one child containing both labels, so no two-leaf tree is consistent. The gain criterion selects $\mathtt{Outlook}$ as the root, but it does not reduce the leaf count here: splitting on $\mathtt{Wind}$ first and then on $\mathtt{Outlook}$ inside the Weak child also yields a three-leaf consistent tree. The greedy ordering is therefore a heuristic for finding a small tree, not a guarantee of finding the smallest one; Proposition 9.2.1 shows that no polynomial-time algorithm can guarantee the smallest consistent tree unless P$\,=\,$NP. What the example does illustrate is the mechanism that makes the heuristic useful: each split is chosen to reduce impurity as much as possible at the current node, and on this sample the resulting tree is already leaf-optimal.

# Overfitting and the Bias-Complexity Dilemma

An unconstrained tree can always drive $\widehat{R}_S(h_T) = 0$ on its training set $S$: grow it until every leaf holds a single training point (or a set of points sharing one label). This is the tree-induction analogue of the memorization hypothesis from Chapter 2 — it exhibits the same overfitting behavior, with tree size (roughly, the number of leaves) playing the role that polynomial degree $M$ played in the curve-fitting example, or that $|\mathcal{H}|$ played in the generalization bound of Chapter 6. Two standard remedies, both instances of the Structural Risk Minimization principle from Chapter 6:

* **Early stopping.** Halt the recursive splitting once a stopping criterion is met (maximum depth, minimum node size, or a minimum required $\mathrm{Gain}$), directly capping the tree's complexity during growth.
* **Pruning.** Grow a large tree first, then remove (merge) subtrees whose removal does not substantially hurt performance on a held-out validation set — trading a controlled increase in the empirical risk on the training sample for a reduction in complexity, in the hope of decreasing $R(h_T)$.

Consistent with the fundamental theorem of statistical learning (Chapter 6), the capacity of $\mathcal{H}_{\text{tree}}$ can also be bounded formally.

**Proposition 9.4.1 (VC dimension of size-limited trees).** Let $\mathcal{H}_{\text{tree},L} \subset \mathcal{H}_{\text{tree}}$ be the subclass of binary trees with axis-aligned threshold splits on $d$ real-valued features and at most $L$ leaves, predicting in $\{0,1\}$. Then
$$
\begin{aligned}
\tau_{\mathcal{H}_{\text{tree},L}}(m) &\;\leq\; \underbrace{C_{L-1}}_{\text{tree shapes}} \cdot \underbrace{\big(d(m+1)\big)^{L-1}}_{\text{splits}} \cdot \underbrace{2^{L}}_{\text{leaf labels}}, \\
&\qquad\text{hence}\qquad d_{VC}(\mathcal{H}_{\text{tree},L}) = O\big(L\log(Ld)\big),
\end{aligned}
$$
where $C_{k} \leq 4^{k}$ is the $k$-th Catalan number, counting the shapes of a binary tree with $k$ internal nodes.

***Proof.*** Fix $m$ points. A tree with at most $L$ leaves has at most $L-1$ internal nodes, and its restriction to the sample is determined by three independent choices:

* the **shape** of the tree, i.e. which binary tree with $L-1$ internal nodes it is: $C_{L-1} \leq 4^{L-1}$ possibilities;
* the **splitting rule** at each internal node. A rule $\mathds{1}(x_j \leq t)$ is determined, *as far as its behavior on the sample is concerned*, by the feature index $j \in [d]$ and by the position of $t$ among the $m$ observed values of feature $j$ — at most $m+1$ distinct positions. So at most $d(m+1)$ effective rules per node, and at most $\big(d(m+1)\big)^{L-1}$ jointly;
* the **label** $c_u \in \{0,1\}$ of each of the at most $L$ leaves: at most $2^L$ possibilities.

Multiplying gives the stated bound on the growth function. Shattering $m$ points requires $2^m \leq \tau_{\mathcal{H}_{\text{tree},L}}(m)$, i.e.
$$m\log 2 \;\leq\; (L-1)\log 4 + (L-1)\log\big(d(m+1)\big) + L\log 2,$$
so $m = O\big(L\log(d) + L\log m\big)$, which forces $m = O\big(L\log(Ld)\big)$ $\square$

Complexity therefore grows essentially linearly with the number of leaves — up to a logarithmic factor for choosing which feature and threshold to split on at each internal node — a formal restatement of the intuition that more leaves means more capacity to overfit. The proof is also worth noting for what it does *not* count: the thresholds $t$ range over an uncountable set, yet only $m+1$ of them are distinguishable on a given sample. This is the same "infinite class, finite restriction" phenomenon that Definition 6.4.1 was introduced to capture.

# PAC Analysis of Decision Trees

**$\mathcal{H}_{\text{tree}}$ itself is not PAC learnable.** Since a tree can grow arbitrarily many leaves, $d_{VC}(\mathcal{H}_{\text{tree},L}) \to \infty$ as $L \to \infty$, hence $d_{VC}(\mathcal{H}_{\text{tree}}) = \infty$ (a tree with $2^k$ leaves can, on real-valued features, always be grown to shatter any $k$ points by isolating each into its own leaf). By Theorem 6.4.6, an infinite VC dimension rules out PAC learnability for the *unconstrained* class: no sample size $m$ guarantees good generalization uniformly over all of $\mathcal{H}_{\text{tree}}$, matching the memorization behavior already noted above. Just as with $k$-nearest neighbors (Chapter 4) and unconstrained polynomial regression (Chapter 2), the fix is not to abandon the hypothesis class, but to control which part of it we search — exactly the role that early stopping and pruning play informally, and which we now make precise.

**Fixed leaf budget: a standard agnostic PAC bound.** Fix a leaf budget $L$ and restrict to $\mathcal{H}_{\text{tree},L}$, which has finite VC dimension $d_L := d_{VC}(\mathcal{H}_{\text{tree},L}) = O(L\log(Ld))$. The Fundamental Theorem of Statistical Learning (Theorem 6.4.7) then applies directly: $\mathcal{H}_{\text{tree},L}$ is agnostic PAC learnable, and with probability at least $1-\delta$, every tree $h \in \mathcal{H}_{\text{tree},L}$ satisfies

$$R(h) \leq \widehat{R}_S(h) + O\left(\sqrt{\frac{L\log(Ld) + \log(1/\delta)}{m}}\right).$$

This already formalizes the early-stopping heuristic: fixing a maximum depth (hence an upper bound on $L$) in advance and running the greedy growth procedure within $\mathcal{H}_{\text{tree},L}$ inherits this generalization guarantee, since the bound holds uniformly over the class. Exact ERM over $\mathcal{H}_{\text{tree},L}$ is NP-hard by Proposition 9.2.1; the greedy procedure is the tractable surrogate that replaces it.

**Unknown leaf budget: decision trees as an instance of SRM.** In practice we do not fix $L$ in advance — the greedy algorithm decides how many leaves to grow from the data itself, effectively searching $\mathcal{H}_{\text{tree}} = \bigcup_{L=1}^\infty \mathcal{H}_{\text{tree},L}$. This is precisely the nonuniform learning setting of Theorem 6.4.12: assign each subclass $\mathcal{H}_{\text{tree},L}$ a weight $w(L)$ with $\sum_{L=1}^\infty w(L) < 1$ (e.g. $w(L) := 6/(\pi^2 L^2)$, or a prefix-free code on $L$ as in the MDL construction of Lemma 6.4.13), and Theorem 6.4.12 gives, with probability at least $1-\delta$, simultaneously for **every** tree $h \in \mathcal{H}_{\text{tree}}$ of any size $L(h)$,

$$R(h) \leq \widehat{R}_S(h) + O\left(\sqrt{\frac{L(h)\log(L(h)d) + \log(1/(w(L(h))\delta))}{m}}\right).$$

The resulting $h_{SRM}$ rule — minimize the right-hand side jointly over tree structure and size — makes explicit what pruning and early stopping approximate: a term that strictly decreases with $L$ (the greedy algorithm's $\widehat{R}_S$, which can always be driven towards $0$ by adding leaves) traded off against a term that strictly increases with $L$ (the complexity penalty $\sqrt{L\log(Ld)/m}$). A pruning rule that stops merging leaves once further merges would raise a held-out validation error is, in effect, estimating this trade-off directly from data rather than from the closed-form penalty above — a data-driven proxy for the same SRM objective.

# Implementation: Growing a Tree on a Real Data Set

Let us implement the greedy tree-growing algorithm of this chapter from scratch — the Gini impurity, the exhaustive split search, and the recursive growth procedure, all exactly as derived above — and test it on the **Wisconsin Breast Cancer** data set: $569$ real tumor samples, each with $30$ real-valued measurements of a cell nucleus, labeled malignant or benign. Unlike the Naive Bayes example of Chapter 8, decision trees natively handle real-valued features via threshold splits, so no discretization is needed here.

```python
import torch as th
from sklearn.datasets import load_breast_cancer

# The only library-provided piece is the raw data set itself; the
# splitting, growth, and prediction logic are all implemented below.
data = load_breast_cancer()
X_all = th.tensor(data.data, dtype=th.float64)
y_all = th.tensor(data.target, dtype=th.long)

th.manual_seed(0)
idx = th.randperm(y_all.numel())
n_train = int(0.7 * y_all.numel())
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

def gini(labels, num_classes=2):
    # G(S) = 1 - sum_c p_c^2
    if labels.numel() == 0:
        return th.tensor(0.0)
    counts = th.bincount(labels, minlength=num_classes).double()
    p = counts / labels.numel()
    return 1.0 - th.sum(p ** 2)

def best_split(X, y):
    # Exhaustively search every (feature, threshold) candidate and keep
    # the one with the highest information gain, Gain(S_v, phi_v).
    m, d = X.shape
    parent_impurity = gini(y)
    best_gain, best_feat, best_thresh = 0.0, None, None
    for j in range(d):
        values = th.unique(X[:, j])
        if values.numel() < 2:
            continue
        # midpoints between consecutive values
        thresholds = (values[:-1] + values[1:]) / 2
        for t in thresholds:
            left_mask = X[:, j] <= t
            n_left = left_mask.sum().item()
            n_right = m - n_left
            if n_left == 0 or n_right == 0:
                continue
            weighted_impurity = (n_left / m) * gini(y[left_mask]) + \
                                 (n_right / m) * gini(y[~left_mask])
            gain = (parent_impurity - weighted_impurity).item()
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, j, t.item()
    return best_feat, best_thresh, best_gain

class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, prediction=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right = left, right
        self.prediction = prediction  # set only for leaves

def build_tree(X, y, depth=0, max_depth=3, min_leaf_size=10):
    majority = th.bincount(y, minlength=2).argmax().item()
    # Early-stopping criteria: depth cap, minimum leaf size, or purity.
    if depth == max_depth or y.numel() < min_leaf_size or gini(y).item() < 1e-12:
        return Node(prediction=majority)
    feat, thresh, gain = best_split(X, y)
    if feat is None or gain <= 1e-12:
        return Node(prediction=majority)
    left_mask = X[:, feat] <= thresh
    left = build_tree(X[left_mask], y[left_mask],
                       depth + 1, max_depth, min_leaf_size)
    right = build_tree(X[~left_mask], y[~left_mask],
                        depth + 1, max_depth, min_leaf_size)
    return Node(feature=feat, threshold=thresh, left=left, right=right)

def predict_one(node, x):
    while node.prediction is None:  # route x from the root to a leaf
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction

def predict(tree, X):
    return th.tensor([predict_one(tree, x) for x in X])

tree = build_tree(X_train, y_train, max_depth=3)

train_preds = predict(tree, X_train)
test_preds = predict(tree, X_test)
train_acc = (train_preds == y_train).double().mean().item()
test_acc = (test_preds == y_test).double().mean().item()

print(f"Decision tree train accuracy: {train_acc:.3f}")
print(f"Decision tree test accuracy:  {test_acc:.3f}")
```

    Decision tree train accuracy: 0.965
    Decision tree test accuracy:  0.918

A tree with at most $2^3=8$ leaves ($\texttt{max\_depth=3}$) already reaches $91.8\%$ test accuracy — close to its $96.5\%$ training accuracy, indicating that the depth cap from Section "PAC Analysis of Decision Trees" above is doing its job of controlling $L$ (and hence $d_{VC}(\mathcal{H}_{\text{tree},L})$) without starving the tree of the capacity it needs to separate the two classes.

Shallow trees are also the standard building blocks of the ensemble methods of Chapter 12: averaging bagged trees gives random forests, and boosting combines depth-one trees (stumps), both as classifiers in AdaBoost and as regression stumps in gradient boosting.
