Lagrangian Propagation Graph Neural Network
===========================================

This repo contains a PyTorch implementation of the LP-GNN model.


- **Conference paper (ECAI 2020)** http://ebooks.iospress.nl/publication/55057
    - **Authors:** Matteo Tiezzi, Giuseppe Marra, Stefano Melacci, Marco Maggini, Marco Gori
    - Link pdf: https://arxiv.org/abs/2002.07684

- **Deep LPGNN -  IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** https://doi.org/10.1109/TPAMI.2021.3073504
    - **Authors:** Matteo Tiezzi, Giuseppe Marra, Stefano Melacci, Marco Maggini


Usage
-----

Requirements
^^^^^^^^^^^^
The LPGNN framework requires the packages **tensorflow**, **numpy**, **scipy**.


To install the requirements you can use the following command
::


      pip install -r requirements.txt


Examples
^^^^^^^^
The **main_subgraph.py** file contains an example on the model usage on several toy datasets (Subgraph matching and Clique detection, with varying sizes of graphs) in an inductive setting.

The **main_chain.py**, **main_enkarate.py**, **main_coradgl.py** shows the usage of the LP-GNN in a semisupervised scenario.
The **main_coradgl.py** example exploits the Deep Grgaph Library Datasets, hence requiring the DGL installation https://www.dgl.ai/pages/start.html

Citing
------

To cite the shallow LP-GNN implementation please use the following publication (Note: ECAI2020 DOI: 10.3233/FAIA200262) ::

    Tiezzi, Matteo, et al. "A Lagrangian Approach to Information Propagation in Graph Neural Networks." arXiv preprint arXiv:2002.07684 (2020).

Bibtex::

    @article{tiezzi2020lagrangian,
      title={A Lagrangian Approach to Information Propagation in Graph Neural Networks},
      author={Tiezzi, Matteo and Marra, Giuseppe and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      journal={arXiv preprint arXiv:2002.07684},
      year={2020}
    }

To cite the Deep LP-GNN please use the following publication::

    Tiezzi, Matteo, et al. "Deep Constraint-based Propagation in Graph Neural Networks." IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

Bibtex::

    @ARTICLE{9405452,
  author={M. {Tiezzi} and G. {Marra} and S. {Melacci} and M. {Maggini}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Deep Constraint-based Propagation in Graph Neural Networks}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3073504}}

Acknowledgment
--------------
We thank Pedro Henrique da Costa Avelar (https://github.com/phcavelar) for the preliminary PyTorch implementation (starting from our original Tensorflow 1.* version).


This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).


License
-------

Released under the 3-Clause BSD license (see `LICENSE.txt`)::

   Copyright (C) 2004-2020 Matteo Tiezzi
   Matteo Tiezzi <mtiezzi@diism.unisi.it>
