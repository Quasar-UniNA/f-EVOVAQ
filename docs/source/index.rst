fast-EVOVAQ's documentation
======================
fast-EVOlutionary algorithms-based toolbox for VAriational Quantum circuits (EVOVAQ) is a novel evolutionary framework designedmto easily train variational quantum circuits through evolutionary techniques on GPUs, and to have a simple interface between these algorithms and quantum libraries, such as Qiskit and Pennylane.

**Optimizers in f-EVOVAQ:**

* Genetic Algorithm

* Differential Evolution

* Memetic Algorithm

* Big Bang Big Crunch

* Particle Swarm Optimization

* CHC Algorithm

* Hill Climbing

Installation
======================
You can install f-EVOVAQ via ``pip``:

.. code-block:: bash

  pip install fevovaq

Pip will handle all dependencies automatically and you will always install the latest version.

Credits
======================
If you use f-EVOVAQ in your work, please cite the following paper:

BibTeX Citation
----------------

.. code-block:: bibtex

   @article{f-evovaq,
     title={f-EVOVAQ: A GPU-based Framework for Evolutionary Training of Variational Quantum Algorithms},
     author={Acampora, Giovanni and Chiatto, Angela and Vitiello, Autilia},
     journal={Accepted to 2026 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
     year={2026},
     publisher={IEEE}}

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials_trainVQCs

.. toctree::
   :hidden:
   :caption: API Guide

   problem
   algorithms
   tools


Indices
******************

* :ref:`genindex`
* :ref:`modindex`
