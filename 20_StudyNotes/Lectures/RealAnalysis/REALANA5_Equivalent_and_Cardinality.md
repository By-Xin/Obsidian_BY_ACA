#RealAnlaysis

Intuitively, we want to 'count' if two sets have the same 'amount' of elements. Thus we have to find a bijection between the two sets.

***Definition (Mapping)***: Given two non-empty sets $A$ and $B$, a mapping $f$ from $A$ to $B$ is a rule that assigns to each element $a\in A$ a related element $b = f(a)\in B$. We write $f:A\to B$ to indicate that $f$ is a mapping from $A$ to $B$. 
- $a$ is called the **pre-image** of $b$ under $f$, and $b$ is called the **image** of $a$ under $f$. 

$\diamond$

***Definition (Injective, Surjective, Bijective)***: A mapping $f:A\to B$ is
- **Injective** (单射) if $x\neq y$ implies $f(x)\neq f(y)$. (i.e. different inputs have different outputs)
- **Surjective** (or **onto**, 满射) $\forall y\in B, \exists x\in A$ such that $f(x) = y$. (i.e. every element in $B$ has a pre-image in $A$)
- **Bijective** (or **one-to-one**, 双射) if it is both injective and surjective.

$\diamond$

***Definition (Equivalent Sets)***: Two sets $A$ and $B$ are **equivalent** (or **equipotent**) if there exists a bijective mapping $f:A\to B$, denoted by $A\sim B$. 
- Especially, define empty set is equivalent to itself $\emptyset\sim\emptyset$.

***Lemma***: Equipotent is an **equivalence relation** on the class of all sets, i.e. it satisfies:
- **Reflexive**: $A\sim A$.
- **Symmetric**: If $A\sim B$, then $B\sim A$.
- **Transitive**: If $A\sim B$ and $B\sim C$, then $A\sim C$.

$\diamond$ 

***Corollary***: All sets with the same 'amount' of elements (i.e. equivalent sets) are equivalent.

***Definition (Finite Set)***: A set $A$ is **finite** if $A\sim\{1,2,\ldots,n\}$ for some $n\in\mathbb{N}$, and the number $n$ is called the **cardinality** of $A$, denoted by $|A|$. Especially, the empty set $\emptyset$ is finite with $|\emptyset|=0$.

***Definition (Infinite Set)***: A set $A$ is **infinite** if it is not finite.
- For all infinite sets, there exists a proper subset that is equivalent to itself.
  - **Example 1**: $\mathbb{Z} \sim \{\text{odd integers}\}$.
    - It indicates a set can have a same 'amount' of elements as its proper subset.
  - **Example 2**: $(0,1) \sim \mathbb{R}$.
    - $x \mapsto \tan(\pi(x-\frac{1}{2}))$ is a bijection from $(0,1)$ to $\mathbb{R}$.

$\diamond$

***Definition (Cardinality)***
  - Two sets $A$ and $B$ have the same **cardinality** if $A\sim B$, denoted as $\overline{\overline{A}} = \overline{\overline{B}}$. 
  - If $A$ is NOT equivalent $B$, but $A$ is equivalent to a proper subset of $B$, then $A$ has a **smaller cardinality** than $B$, denoted as $\overline{\overline{A}} < \overline{\overline{B}}$.
  - Sometimes also introduce the notation $\overline{\overline{A}} \leq \overline{\overline{B}}$ to indicate $A$ has a smaller or equal cardinality than $B$.

***Theorem (Bernstein's Theorem)***: If $\overline{\overline{A}} \leq \overline{\overline{B}}$ and $\overline{\overline{B}} \leq \overline{\overline{A}}$, then $\overline{\overline{A}} = \overline{\overline{B}}$.

- **Sketch of Proof**
  - $A: A \stackrel{1:1}{\longrightarrow} \varphi(A) \triangleq B_1 \subset B$. $B: B \stackrel{1:1}{\longrightarrow} \psi(B) \triangleq A_1 \subset A$.
  - $A_1: A_1 \stackrel{1:1}{\longrightarrow} \varphi(A_1) \triangleq B_2 \subset B$. $B_1: B_1 \stackrel{1:1}{\longrightarrow} \psi(B_1) \triangleq A_2 \subset A$. 
  - Continue this process, we have $A \supset A_1 \supset A_2 \supset \ldots$ and $B \supset B_1 \supset B_2 \supset \ldots$.
  - Moreover, $A = (A-A_1)\cup (A_1-A_2)\cup\ldots$ and $A_1 = (A_1-A_2)\cup (A_2-A_3)\cup\ldots$.
- To apply Bernstein's Theorem, to prove $A\sim B$, it is sufficient to construct the following two mappings:
  - $A \stackrel{\text{Injective}}{\longrightarrow} B, B \stackrel{\text{Injective}}{\longrightarrow} A$.
  - Or equivalently, $A \stackrel{\text{Bijective}}{\longrightarrow} B$, $A \stackrel{\text{Onto}}{\longrightarrow} B$.

- ***Example 1***: $(0,1) \sim (0,1]$.
  - Since $(0,1) \mapsto (0,1) \subset (0,1]$; and $(0,1] \mapsto (0,1/2] \subset (0,1)$.Then by Bernstein's Theorem, $(0,1) \sim (0,1]$.

- ***Example 2***: $C \subset B \subset A$, and $A\sim C$. Then $B\sim A$.
  - Since $A\sim C$, there exists a $f: A \stackrel{\text{Injective}}{\longrightarrow} C$. 
  - Plus, since $B\subset A$, there exists a $g: B \stackrel{\text{Injective}}{\longrightarrow} A$ (as long as $b\in B \mapsto b\in A$). Similarly, since $C\subset B$, there exists a $h: C \stackrel{\text{Injective}}{\longrightarrow} B$.
  - Then we have both $A \stackrel{\text{Injective}}{\longrightarrow} B$ and $B \stackrel{\text{Injective}}{\longrightarrow} A$. By Bernstein's Theorem, $A\sim B$.