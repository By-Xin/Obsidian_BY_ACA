Hi Baiying, Your background looks strong, and I am open to all possible collaborations. Welcome aboard.  

- You may first dive into the datasets at [https://virtualcellchallenge.org/](https://virtualcellchallenge.org/ "https://virtualcellchallenge.org/"). We also collect some literature and resources in [https://www.notion.so/Virtual-Cell-Challenge-235678e327d0802fbdf1f05da322c2ef?source=copy_link](https://www.notion.so/Virtual-Cell-Challenge-235678e327d0802fbdf1f05da322c2ef?source=copy_link "https://www.notion.so/Virtual-Cell-Challenge-235678e327d0802fbdf1f05da322c2ef?source=copy_link").

- The datasets are stored in the standard format adata; anndata. 
	```python
import scanpy as sc
adata = sc.read_h5ad("vcc_data/adata_Training.h5ad")    
print(f"Data shape: {adata.shape}")
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

# Load validation data
validation = pd.read_csv("vcc_data/pert_counts_Validation.csv")
print(f"Validation perturbations: {len(validation)}")

# Load gene names
genes = pd.read_csv("vcc_data/gene_names.csv", header=None, names=['gene'])
print(f"Total genes: {len(genes)}")
```
  



- You can try some methods like scGen [https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html](https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html "https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html") . They are easy to implement and should help quickly understand the goal of cell prediction. 


- Zichu and Zichong (cc'd here) are currently helping managing the project. Zichu (子初) is the incoming  5-th year phd student, who has rich experience in perturb-seq data. Zichong is currently senior year and will do phd with me. You can ask them for help if you have any questions. 

- Considering your strength in coding, 
	- you may want to read the literature on mask auto-encoder [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377 "https://arxiv.org/abs/2111.06377"). We may want to add some sparsity to existing methods. I know Yixuan has some sparse mask paper. 
	- You may also read [[TOSICA]]: Transformer for One-Stop Interpretable Cell-type and let' see how we can incorporate the GO-KEGG information into the model. 
	- It would also be nice if we can do transfer learning. 


  
- We have regular meeting on Monday to discuss the recent progress. I can send you the link if you are interested. Prof Jiaye Teng is also welcome. 
