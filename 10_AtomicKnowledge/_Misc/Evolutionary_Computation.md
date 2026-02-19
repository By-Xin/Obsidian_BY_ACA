#EvoluationaryAlgorithm

## Introduction to Evolutionary Algorithms
- Evolutionary Algorithms (EAs): a subset of etaheuristic algorithm inspired by biological evolution, which include Genetic Algorithm,Evolutionary Programming, Evolution Strategies, Differential Evolution.

- Evolutionary Algorithms $\subset$ Metaheuristic Algorithms $\subset$ Heuristic Algorithms (启发式算法) $\subset$ Stochastic Local Search Algorithms $\subset$ Search and Enumeration Algorithms （搜索枚举算法）
- Evolutionary Algorithms unique features: 
    - Population-based: maintain a population of candidate solutions

- Evolutionary Algorithms:
    ```
    X_0 := Initialize Population
    is_stop := False
    iteration := 0
    evaluate_individual_fitness(X_0)

    while (is_stop == False):

        SELECTION: Select parents from population X_t
        VARIATION: Generate offspring individuals by variation operators to parents
        FITNESS_EVALUATION: Evaluate the fitness of offspring individuals
        REPRODUCTION: Generate population X_{t+1} by replacing the old least-fit individuals 

        iteration += 1

        if (termination_condition):
            is_stop = True
            
    return best_individual
    ```


- **Optimization**: find global optimal, making thebalance between exploration and exploitation
  - **Exploration**: Generate new individuals from unsearched regions. Its purpose is to discover potential global optimal solutions, avoiding local optimal solutions.
  - **Exploitation**: Improve the quality of the current solutions within the searched regions. Its purpose is to find the best solution in the known regions.
  - Trade-off: If exploration is too much, it may waste time searching for the global optimal solution; while if exploitation is too much, it may fall into local optimal solutions and converge too early.
  - To balance exploration and exploitation:
    - Variation operators: exploration
    - Selection and reproduction: exploitation


## Key concepts of Evolutionary Algorithms

### Representation

- Representation: encode (represent) a solution as a individual
- Encoding: $\textit{Pheonotype} \rightarrow \textit{Genotype}$
  - Genotype: the most mathematic representation of a solution, the one the algorithm directly manipulates
  - Pheonotype: the actual solution that the genotype represents, the one has more practical meaning
  ![1728116737685](image/EvolutionaryComputation/1728116737685.png)
  ![1728115901253](image/EvolutionaryComputation/1728115901253.png)



- Representation types:
  - Binary representation: $0$ or $1$
  - Real-value representation
  - Random keys representation

- **Representation**: each solution is called anindividual
- **Fitness** (objective function): a measure ofhow good a solution is
- **Variation** operators: mutation, crossover
- **Selection** and **Reproduction**: survivalof the fittest

### Variation Operators

#### Mutation

- Mutation: flip each bit of a binary string with a mutation rate $p_m$
- Defalut mutation rate: $p_m = 1/n$, where $n$ is the number of bits in the binary string. But it can be adjusted according to the problem, e.g., $p_m\in [1/n, 1/2]$
- If $p_m$ is small
  - mutation can be seen as creating a small random perturbation to the solution (parent genotype)
  - Offspring largely similar to the parent, having a small Hamming distance in genotype space
- Mutation is actually doing a **local search** : **exploit** current solution by randomly **explore** the neighborhood of the current solution

#### Crossover

- Crossover: combine two parent genotypes to generate offspring genotypes

![1728117755031](image/EvolutionaryComputation/1728117755031.png)