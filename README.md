# Classy Ensemble
Classy Ensemble: A Novel Ensemble Algorithm for Classification

Code accompanying the paper: [M. Sipper, "Classy Ensemble: A Novel Ensemble Algorithm for Classification"](https://arxiv.org/abs/2302.10580).

* `main.py`: main module for running ML models
* `main_dl.py`: main module for running DL models
* `main_dl_s.py`: main module for running DL models / load network outputs from saved files
* `main_cleen.py`: main module for Classy Evolutionary Ensemble
* `basic_ensemble.py`: class BasicEnsemble -- basic ensemble functionality
* `classy_ensemble.py`: class ClassyEnsemble -- the Classy Ensemble algorithm
* `classy_pre.py`: class ClassyEnsemble -- Classy Ensemble that uses preloaded outputs, used by main_cleen (Classy Evolutionary Ensemble)
* `cluster_ensemble.py`: class ClusterEnsemble -- generates ensemble through cluster-based pruning
* `classy_cluster.py`: class ClassyClusterEnsemble -- a combination of classy ensemble and cluster-based pruning
* `classy_cluster_v2.py`: class ClassyClusterEnsemble, version 2 (refer to paper)
* `lexigarden.py`: class Lexigarden -- generates ensemble through lexigarden algorithm
* `networks.py`: deep networks
* `generate_outputs.py`: generate output vectors for pretrained models
* `torch-datasets.py`: tourchvision datasets
* `ml_models.py`: ML models used 
* `utils.py`: various utlity functions
* `stats.py`: some statistics for paper


### Citation

Citations are always appreciated 😊:
```
@article{sipper2023classy,
    author = {Sipper, Moshe},
    title = {Classy Ensemble: A Novel Ensemble Algorithm for Classification},
    publisher = {arXiv},
    year = {2023},
    url = {https://arxiv.org/abs/2302.10580},
}
```
