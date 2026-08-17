
## Current Tasks and Challenges

Such classification is not a simple key-word matching problem. It is mainly based on two dimensions: the problem contingent and the utility, which contain very large implicit underlying information. It also reveals that, this 4PT classification framework is a multi-stage, logically sequential process.

During the automation, instruction follow and accuracy are our key concerns. We should utilize the structured coding questions, to break down the sophisticated framework into several, manageable questions, further asking models to give evidence from the original paper. 

## Framework Pathways: API-calling v.s. Self-deployed Models

Basically there are two strategies:
- Using commercial APIs to call the latest most advanced closed-source models
- Deploying and fine-tuning our own LLM using the open-sourced models on a private cloud infrastructure. 
These two paths are not simply about which is better or worse, but rather represent different strategic trade-offs in terms of development speed, cost control, data security, and long-term asset accumulation.

### API Priority

This method's core is to use APIs from third-party service providers (like Anthropic, OpenAI, and Google) to instantly access the world's most advanced large language models.

- **Core Value Proposition**: The powerful capabilities of models like Anthropic Claude, OpenAI GPT, or Google Gemini can be used immediately, without any infrastructure setup or maintenance.

- **Advantages**:
    - **Rapid Prototyping**: A functional classifier can be developed and tested in just a few days or weeks using only **prompt engineering**. This is crucial for quickly validating the feasibility of an automation solution.
    - **Top-Tier Performance**: These closed-source models consistently lead in benchmark tests for various complex reasoning tasks. As analyzed in Section, strong reasoning ability is a prerequisite for successfully completing the 4PT classification task.

- **Disadvantages and Strategic Risks**:
    - **Variable and Uncapped Costs**: Costs are directly tied to usage and billed by the number of tokens processed 11. When the number of papers to be classified scales from a few hundred to tens of thousands, the operational costs will increase linearly and can become extremely high.
    - **Data Privacy and Confidentiality**: Transmitting academic papers that may contain unpublished research or sensitive data to a third-party API provider could conflict with an institution's data security policies or confidentiality agreements. 
        
    - **Vendor Lock-in and Model Depreciation**: The solution becomes deeply tied to a specific vendor's API. Model versions are constantly updated and deprecated (for example, Anthropic's documentation has marked Claude 3.5 Sonnet as "deprecated"), which may force research teams to invest extra time and cost into re-adapting prompts and re-validating performance, affecting the long-term consistency of the research.
        

### **Self-Hosting Method: Building a Sovereign Asset**

This method's core is to select a high-performing open-source model (like Meta's Llama or Mistral AI's Mistral Large) and fine-tune and deploy it using your own labeled data on a private cloud platform (such as Amazon Web Services (AWS) SageMaker or Google Cloud Platform (GCP) Vertex AI).

- **Core Value Proposition**: With a one-time upfront investment, you build a classification tool that is completely under your control, highly customized, and has predictable long-term costs.
    
- **Advantages**:
    
    - **Scalable Cost Control**: After the initial model fine-tuning is complete, subsequent inference costs are primarily the fixed rental fees for cloud servers and are largely independent of the number of papers processed 15. For large-scale or continuous classification tasks, this model is far more cost-effective in the long run than API calls.
        
    - **Data Sovereignty**: All papers to be classified and labeled data remain within your own secure cloud environment, completely eliminating third-party data privacy and confidentiality concerns.
        
    - **Creation of a Lasting Asset**: The fine-tuned model itself becomes a valuable intellectual property. It is a specialized tool precisely tailored to the 4PT framework, representing a programmatic embodiment of the research methodology that can be used stably for the long term.
        
- **Disadvantages and Upfront Investment**:
    
    - **Engineering Overhead**: Require some knowledge of cloud computing (AWS/GCP) and Machine Learning Operations (MLOps) to set up, configure, and maintain the training and inference environments.
        
    - **Initial Time and Financial Investment**: The model fine-tuning process requires renting expensive servers with high-performance GPUs in the short term, which constitutes a significant one-time upfront cost.
        
    - **Model Selection Risk**: Although the performance of open-source models is catching up rapidly, there might still be a slight gap between their performance and that of the most advanced closed-source models on certain extremely complex reasoning tasks.

### Integration of Both Methods

Actually these two methods are not contradictory to each other. API is more rapid, easy-to-deploy, and cost relatively low for small datasets.  While self-host is the long-term large-scale final solution (if needed).

In the first stage, the key problem is: **whether LLM is capable to understand and properly classify the 4PT framework.** If we managed to achieve a satisfying initial outcome with top-tier APIs and carefully designed prompts (e.g. an accurate rate at 85%), then we are at least convinced that such problem is solvable. Moreover, during this process, we can greatly enlarge our data from LLM, which provide more information for future local deployment.

Furthermore, if we have more requirements w.r.t. data security or self-deployment, we may then transform the API to local models. But generally it may not be a first choice, as its great initial cost. 