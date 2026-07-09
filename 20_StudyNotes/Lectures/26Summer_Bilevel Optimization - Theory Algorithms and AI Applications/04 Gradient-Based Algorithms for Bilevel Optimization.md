# Gradient-Based Algorithms for Bilevel Optimization

## 0. Overall Storyline and Motivation
(What the hypergradient is and why we need it; comparison of the two technical routes)

## 1. Unconstrained Bilevel Problem and Failure of the Naive Method
### 1.1 Problem Setup: min F s.t. y ∈ argmin f
### 1.2 Alternating Update (LL / UL step)
### 1.3 Counterexample: Why Alternating Update Fails
      —— Key: it loses the sensitivity of y*(x) to x

## 2. Strongly Convex Lower Level: Implicit Gradient Methods
### 2.1 Reduced Problem and the Chain Rule
### 2.2 Implicit Function Theorem → Hypergradient Formula
### 2.3 Two Computational Costs (approximating y*, inverse-Hessian-vector product)
### 2.4 Classical Double-Loop Approximation (CG / Neumann truncation)

## 3. Single-Loop Implicit Gradient Method
### 3.1 Tracking the Inverse-Hessian Product via an Auxiliary Variable v_k
### 3.2 Joint Update of (y, v, x)
### 3.3 Attempt to Remove Strong Convexity: Lower-Level Regularization
### 3.4 Counterexample: Regularization = Minimum-Norm Selection, Not Necessarily the True Solution

## 4. BAMM: Averaged Method of Multipliers without Lower-Level Strong Convexity
### 4.1 Optimistic Viewpoint and the Aggregation Function φ_μ
### 4.2 MPEC / KKT Reformulation
### 4.3 Stationarity Measure without Strong Convexity (KKT Residual)
### 4.4 The sl-BAMM Algorithm (Four-Step Update)
### 4.5 Convergence Theorem and Parameter Choices
### 4.6 Lyapunov Function and Proof Roadmap

## 5. Iterative Differentiation Methods (ITD)
### 5.1 Differentiating the Lower-Level Trajectory: Basic Idea
### 5.2 Forward Mode (sensitivity Z_t)
### 5.3 Reverse Mode (adjoint q_t)
### 5.4 Comparison with Implicit Gradient Methods
### 5.5 Hypergradient Convergence and Bias O(ρ^T)
### 5.6 Non-Unique Solutions: Counterexample and BDA (Descent Aggregation)
### 5.7 Non-Convex Lower Level: IAPTT-GM (Initialization Auxiliary + Pessimistic Truncation)
### 5.8 Experiments and Applications (data hyper-cleaning, few-shot classification)

## References