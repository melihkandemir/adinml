Every hypothesis class we have used so far predicts through a fixed algebraic form: a dot product for linear predictors (Chapter 2), a sigmoid or softmax of a dot product for classifiers (Chapter 3), a weighted sum of similarities for kernel methods (Chapter 9). A **decision tree** [Breiman et al. (1984)](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)<!-- cite: breiman1984cart | book | author={Breiman, Leo and Friedman, Jerome H. and Olshen, Richard A. and Stone, Charles J.}; title={Classification and Regression Trees}; publisher={Wadsworth}; year={1984} --> predicts through a different mechanism entirely: a sequence of "if-then" questions about individual features, organized in a tree, that recursively partitions the feature space $\mathcal{X}$ into regions, each assigned a single output.

# The Hypothesis Class

A decision tree is a rooted tree in which:

* every **internal node** $v$ is labeled with a **splitting rule** $\phi_v: \mathcal{X} \rightarrow \{1,\ldots,k_v\}$ that routes an input $x$ to one of $v$'s $k_v$ children — for the binary trees we focus on here, $k_v=2$ and $\phi_v(x) := \mathds{1}(x_j \leq t)$ for some feature index $j$ and threshold $t$ (or $\phi_v(x) := \mathds{1}(x_j = a)$ for a categorical feature taking the value $a$),
* every **leaf** $u$ is labeled with a prediction $c_u \in \mathcal{Y}$.

A tree $T$ computes a hypothesis $h_T: \mathcal{X} \rightarrow \mathcal{Y}$ by routing $x$ from the root, following $\phi_v(x)$ at each internal node $v$, until it reaches a leaf $u$, and outputting $h_T(x) := c_u$. Since each leaf's region is exactly the intersection of the splitting rules on the path from the root, the collection of leaves of $T$ partitions $\mathcal{X}$ into disjoint axis-aligned regions, and $h_T$ is piecewise constant on this partition. The **hypothesis class of decision trees**, $\mathcal{H}_{\text{tree}}$, is the union of $h_T$ over all finite binary trees $T$ built from the allowed splitting rules — an infinite, nonuniform hypothesis class in the sense of Chapter 5, since larger trees induce finer partitions and hence more expressive hypotheses.

Formally, then, a decision tree instantiates the general recursive partitioning scheme

$$\begin{gathered}
h_T(x) = \sum_{u \in \mathrm{leaves}(T)} c_u \cdot \mathds{1}\big(x \in R_u\big), \\
R_u := \bigcap_{v \in \mathrm{path}(u)} \phi_v(x)\text{-consistent half-space at } v,
\end{gathered}$$

where $\mathrm{path}(u)$ is the sequence of internal nodes from the root to leaf $u$, and $\{R_u\}_{u \in \mathrm{leaves}(T)}$ partitions $\mathcal{X}$.

# Growing a Tree: A Greedy ERM Surrogate

Given a training set $S$, ERM over $\mathcal{H}_{\text{tree}}$ would search over all trees for the one minimizing $\widehat{R}_S(h_T)$. This is computationally intractable — even fixing the tree's shape, choosing the best splits jointly is NP-hard. In practice, we build the tree by a **greedy, recursive** procedure:

1. Start with a single leaf holding all of $S$.
2. At each leaf currently holding a subset $S_v \subseteq S$, consider splitting it into two children by some rule $\phi_v$. Among all candidate rules (all features, all thresholds), pick the one that most reduces an **impurity** measure of the label distribution within the node (defined below).
3. Recurse into each child with its corresponding subset of $S_v$, and repeat, until a stopping criterion is met (e.g. a maximum depth, a minimum leaf size, or zero remaining impurity).
4. Label each final leaf $u$ with the majority class of $S_v$ restricted to that leaf (classification) or the mean label (regression).

The key design choice is the **impurity measure**, which stands in for $\widehat{R}_S(h)$ as a differentiable-in-spirit, locally computable proxy that the greedy procedure can optimize one split at a time. For a node $v$ holding a subset $S_v$ of size $m_v := |S_v|$, with class proportions $\widehat{p}_c := \frac{|\{(x,y) \in S_v : y=c\}|}{m_v}$ for $c \in \{1,\ldots,C\}$, two common choices are:

* **Entropy** (of the empirical label distribution at the node): $H(S_v) := -\sum_{c=1}^C \widehat{p}_c \log \widehat{p}_c$, borrowed directly from information theory — it is maximized when $S_v$'s labels are uniformly spread across classes (there is the most "uncertainty" about the label at this node) and zero when $S_v$ is pure (a single class). Following the convention of these notes, $\log$ here is the natural logarithm, so entropy is measured in nats rather than the more familiar bits; the base only rescales $H$ by a positive constant and therefore never changes which split is selected.
* **Gini index**: $G(S_v) := 1 - \sum_{c=1}^C \widehat{p}_c^2 = \sum_{c=1}^C \widehat{p}_c(1-\widehat p_c)$, the probability that two labels drawn independently (with replacement) from $S_v$'s empirical distribution disagree. It behaves similarly to entropy — zero for a pure node, maximal for a uniform one — but is cheaper to compute and does not involve a logarithm.

Given an impurity measure $\mathrm{Imp}(\cdot) \in \{H, G\}$ and a candidate split of $S_v$ into $S_{v,\text{left}}$ and $S_{v,\text{right}}$, the **information gain** of the split is the reduction in impurity, weighted by the resulting subset sizes:

$$\mathrm{Gain}(S_v, \phi_v) := \mathrm{Imp}(S_v) - \frac{|S_{v,\text{left}}|}{m_v}\mathrm{Imp}(S_{v,\text{left}}) - \frac{|S_{v,\text{right}}|}{m_v}\mathrm{Imp}(S_{v,\text{right}}).$$

At each node, the greedy algorithm picks the split $\phi_v$ maximizing $\mathrm{Gain}(S_v,\phi_v)$ over all candidate features and thresholds. This greedy, node-by-node maximization is what makes decision tree induction tractable: it never revisits an earlier split, and never jointly optimizes two splits together, at the cost of no longer being guaranteed to find the ERM-optimal tree of a given size.

# Overfitting and the Bias-Complexity Dilemma

An unconstrained tree can always drive $\widehat{R}_S(h_T) = 0$ on its training set $S$: grow it until every leaf holds a single training point (or a set of points sharing one label). This is the tree-induction analogue of the memorization hypothesis from Chapter 1 — it exhibits the same overfitting behavior, with tree size (roughly, the number of leaves) playing the role that polynomial degree $M$ played in the curve-fitting example, or that $|\mathcal{H}|$ played in the generalization bound of Chapter 5. Two standard remedies, both instances of the Structural Risk Minimization principle from Chapter 5:

* **Early stopping.** Halt the recursive splitting once a stopping criterion is met (maximum depth, minimum node size, or a minimum required $\mathrm{Gain}$), directly capping the tree's complexity during growth.
* **Pruning.** Grow a large tree first, then remove (merge) subtrees whose removal does not substantially hurt performance on a held-out validation set — trading a controlled increase in the empirical risk on the training sample for a reduction in complexity, in the hope of decreasing $R(h_T)$.

Consistent with the fundamental theorem of statistical learning (Chapter 5), the capacity of $\mathcal{H}_{\text{tree}}$ can also be bounded formally.

**Proposition 7.1 (VC dimension of size-limited trees).** Let $\mathcal{H}_{\text{tree},L} \subset \mathcal{H}_{\text{tree}}$ be the subclass of binary trees with axis-aligned threshold splits on $d$ real-valued features and at most $L$ leaves, predicting in $\{0,1\}$. Then
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

Complexity therefore grows essentially linearly with the number of leaves — up to a logarithmic factor for choosing which feature and threshold to split on at each internal node — a formal restatement of the intuition that more leaves means more capacity to overfit. The proof is also worth noting for what it does *not* count: the thresholds $t$ range over an uncountable set, yet only $m+1$ of them are distinguishable on a given sample. This is the same "infinite class, finite restriction" phenomenon that Definition 5.9 was introduced to capture.

# PAC Analysis of Decision Trees

**$\mathcal{H}_{\text{tree}}$ itself is not PAC learnable.** Since a tree can grow arbitrarily many leaves, $d_{VC}(\mathcal{H}_{\text{tree}}) = \lim_{L\to\infty} d_{VC}(\mathcal{H}_{\text{tree},L}) = \infty$ (a tree with $2^k$ leaves can, on real-valued features, always be grown to shatter any $k$ points by isolating each into its own leaf). By Theorem 5.3, an infinite VC dimension rules out PAC learnability for the *unconstrained* class: no sample size $m$ guarantees good generalization uniformly over all of $\mathcal{H}_{\text{tree}}$, matching the memorization behavior already noted above. Just as with $k$-nearest neighbors (Chapter 3) and unconstrained polynomial regression (Chapter 1), the fix is not to abandon the hypothesis class, but to control which part of it we search — exactly the role that early stopping and pruning play informally, and which we now make precise.

**Fixed leaf budget: a standard agnostic PAC bound.** Fix a leaf budget $L$ and restrict to $\mathcal{H}_{\text{tree},L}$, which has finite VC dimension $d_L := d_{VC}(\mathcal{H}_{\text{tree},L}) = O(L\log(Ld))$. The Fundamental Theorem of Statistical Learning (Theorem 5.4) then applies directly: $\mathcal{H}_{\text{tree},L}$ is agnostic PAC learnable, and with probability at least $1-\delta$, every tree $h \in \mathcal{H}_{\text{tree},L}$ satisfies

$$R(h) \leq \widehat{R}_S(h) + O\left(\sqrt{\frac{L\log(Ld) + \log(1/\delta)}{m}}\right).$$

This already formalizes the early-stopping heuristic: fixing a maximum depth (hence an upper bound on $L$) in advance and running ERM (i.e. the greedy growth procedure, as a tractable surrogate for exact ERM) within $\mathcal{H}_{\text{tree},L}$ inherits this generalization guarantee.

**Unknown leaf budget: decision trees as an instance of SRM.** In practice we do not fix $L$ in advance — the greedy algorithm decides how many leaves to grow from the data itself, effectively searching $\mathcal{H}_{\text{tree}} = \bigcup_{L=1}^\infty \mathcal{H}_{\text{tree},L}$. This is precisely the nonuniform learning setting of Theorem 5.7: assign each subclass $\mathcal{H}_{\text{tree},L}$ a weight $w(L)$ with $\sum_{L=1}^\infty w(L) < 1$ (e.g. $w(L) := 6/(\pi^2 L^2)$, or a prefix-free code on $L$ as in the MDL construction of Lemma 5.3), and Theorem 5.7 gives, with probability at least $1-\delta$, simultaneously for **every** tree $h \in \mathcal{H}_{\text{tree}}$ of any size $L(h)$,

$$R(h) \leq \widehat{R}_S(h) + O\left(\sqrt{\frac{L(h)\log(L(h)d) + \log(1/(w(L(h))\delta))}{m}}\right).$$

The resulting $h_{SRM}$ rule — minimize the right-hand side jointly over tree structure and size — makes explicit what pruning and early stopping approximate: a term that strictly decreases with $L$ (the greedy algorithm's $\widehat{R}_S$, which can always be driven towards $0$ by adding leaves) traded off against a term that strictly increases with $L$ (the complexity penalty $\sqrt{L\log(Ld)/m}$). A pruning rule that stops merging leaves once further merges would raise a held-out validation error is, in effect, estimating this trade-off directly from data rather than from the closed-form penalty above — a data-driven proxy for the same SRM objective.

# Illustrative Example

Consider a tiny data set for deciding whether to play outdoor tennis, with two binary features — $\mathtt{Outlook} \in \{\mathrm{Sunny}, \mathrm{Rainy}\}$ and $\mathtt{Wind} \in \{\mathrm{Weak}, \mathrm{Strong}\}$ — and label $\mathtt{Play} \in \{\mathrm{Yes}, \mathrm{No}\}$:

| $\mathtt{Outlook}$ | $\mathtt{Wind}$ | $\mathtt{Play}$ |
|---|---|---|
| Sunny | Weak | Yes |
| Sunny | Strong | Yes |
| Sunny | Weak | Yes |
| Sunny | Strong | No |
| Rainy | Weak | No |
| Rainy | Weak | No |
| Rainy | Strong | No |
| Rainy | Strong | No |

Here $m=8$, with $4$ "Yes" and $4$ "No", so the root's Gini index is $G(S) = 1 - (\tfrac12)^2 - (\tfrac12)^2 = 0.5$.

**Candidate split on $\mathtt{Outlook}$.** This sends the $4$ Sunny points (3 Yes, 1 No) to the left child and the $4$ Rainy points (0 Yes, 4 No) to the right child:

$$\begin{gathered}
G(S_{\text{left}}) = 1 - \big(\tfrac34\big)^2 - \big(\tfrac14\big)^2 = 0.375, \\
G(S_{\text{right}}) = 1 - 1^2 - 0^2 = 0.
\end{gathered}$$

$$\mathrm{Gain}(S, \mathtt{Outlook}) = 0.5 - \tfrac{4}{8}(0.375) - \tfrac{4}{8}(0) = 0.5 - 0.1875 = 0.3125.$$

**Candidate split on $\mathtt{Wind}$.** This sends the $4$ Weak points (2 Yes, 2 No) to the left child and the $4$ Strong points (1 Yes, 3 No) to the right child:

$$\begin{gathered}
G(S_{\text{left}}) = 1 - \big(\tfrac12\big)^2 - \big(\tfrac12\big)^2 = 0.5, \\
G(S_{\text{right}}) = 1 - \big(\tfrac14\big)^2 - \big(\tfrac34\big)^2 = 0.375.
\end{gathered}$$

$$\mathrm{Gain}(S, \mathtt{Wind}) = 0.5 - \tfrac48(0.5) - \tfrac48(0.375) = 0.5 - 0.4375 = 0.0625.$$

Since $\mathrm{Gain}(S,\mathtt{Outlook}) = 0.3125 > 0.0625 = \mathrm{Gain}(S,\mathtt{Wind})$, the greedy algorithm splits on $\mathtt{Outlook}$ first. The right child (Rainy) is already pure ($G=0$) and becomes a leaf predicting $\mathrm{No}$. The left child (Sunny, $3$ Yes/$1$ No) is not pure, so we recurse: splitting it further on $\mathtt{Wind}$ sends its single Strong/No point to its own leaf and the remaining $3$ Weak/Yes points to another, both pure, terminating the recursion. The resulting tree is

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

Observe that this tree achieves $\widehat{R}_S(h_T) = 0$ using only $3$ leaves out of the $2^2=4$ possible input combinations having been observed — the greedy split ordering (largest information gain first) is exactly what let it reach a pure partition using the *fewest* splits, which is the practical mechanism by which impurity-driven greedy growth tends to produce small, and hence better-generalizing, trees compared to growing in an arbitrary order.

# Implementation: Growing a Tree on a Real Data Set

Let us implement the greedy tree-growing algorithm of this chapter from scratch — the Gini impurity, the exhaustive split search, and the recursive growth procedure, all exactly as derived above — and test it on the **Wisconsin Breast Cancer** data set: $569$ real tumor samples, each with $30$ real-valued measurements of a cell nucleus, labeled malignant or benign. Unlike the Naive Bayes example of Chapter 6, decision trees natively handle real-valued features via threshold splits, so no discretization is needed here.

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
