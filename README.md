# schreiber-scripts
These are implementations of Amelie Schreiber's awesome HF tutorials (https://huggingface.co/AmelieSchreiber)

- log_likelihoods.py - plotting effects of mutations using the log-likelihood ratios (LLR) of each single point mutation 
- esm2_scoring_variants.py - implementing metrics for log-likelihood ratios, pseudo-perplexity (PPPL), wild-type marginal (wt-marginal), and masked marginal
- evoprot.py - directed evolution of protein sequences with esm2 and evoprotgrad
- esm2ptm.py - qlora for esm2 & post translational modification site prediction
- ppi_interactions.py - use masked language modeling loss to predict protein-protein interactions for a set of all proteins
- ranking_binders.py - ranking potential binding partners for a single protein of interest
