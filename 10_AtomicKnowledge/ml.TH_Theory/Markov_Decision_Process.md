#MarkovDecisionProcess #ReinforcementLearning

## Basic Concepts

- Basic Elements of MDP
  - State (${S}$)
  - Action (${A}$)
  - Reward (${R}$)
  - State Transition Probability ($P$)
  - Discount Factor ($\gamma$)
  - Policy ($\pi : {S} \rightarrow {A}$)

- Initially, the agent starts in a state $s_0$ and selects an action $a_0$ according to the policy $\pi(s_0)$. Then the agent will transist to a new state $s_1 \sim P(s_1|s_0, a_0)$ . Then the agent will do action $a_1$ according to the policy $\pi(s_1)$ and so on. Therefore, according to the states and actions, the
total reward the agent will get is: $R = \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)$. And the goal of the agent is to maximize the total reward, or to say, to learn a policy that maximizes the total reward.

- Value Function for Policy $\pi$: $V^{\pi}(s)$
  - For a given policy $\pi$, the value function $V^{\pi}(s): S \rightarrow \mathbb{R}$ is the expected total reward the agent will get starting from state $s$ and following the policy $\pi$. 
  - Bellman Equation for Value Function
    - $V^{\pi}(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) V^{\pi}(s')$
    - The expected total reward at a given state $s$  is the immediate reward $R(s, \pi(s))$ plus the expected total discounted future reward $\gamma \sum_{s'} P(s'|s, \pi(s)) V^{\pi}(s')$.
    - If we consider all possible states, the Bellman equation can be written in a matrix form: $V^{\pi} = R^{\pi} + \gamma P^{\pi} V^{\pi}$, where $R^{\pi}$ is the reward matrix and $P^{\pi}$ is the state transition probability matrix. And it can be analytically solved as: $V^{\pi} = (I - \gamma P^{\pi})^{-1} R^{\pi}$.

- Optimal Value Function: $V^*(s)$
  - The optimal value function $V^*(s)$ is the maximum value function over all possible policies: $V^*(s) = \max_{\pi} V^{\pi}(s)$.
  - Bellman Optimality Equation for Value Function
    - $V^*(s) = R(s) + \gamma \max_{a} \sum_{s'} P(s'|s, a) V^*(s')$
    - The best expected total reward at a given state $s$ is the immediate reward $R(s)$ plus the action that maximizes the expected total discounted future reward $\gamma \max_{a} \sum_{s'} P(s'|s, a) V^*(s')$.
    - Once the $V^*(s)$ is obtained, the optimal policy can be derived as: $\pi^*(s) = \arg\max_{a} \sum_{s'} P(s'|s, a) V^*(s')$.


- Claim: $ V^*(s) = V^{\pi^*}(s) \ge V^{\pi}(s) \quad \forall s \in S, \forall \pi$

- Value Iteration Algorithm
  - Synchrounous Update:
    ``` 
    Initialize V(s) = 0 for every s
    For every s:
        V(s) <- R(s) + \gamma \max_{a} \sum_{s'} P(s'|s, a) V(s')
    ```
