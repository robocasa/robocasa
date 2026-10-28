.. RoboCasa documentation master file, created by
   sphinx-quickstart on Tue Jun 18 15:47:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RoboCasa for Unitree G1 Evaluation
==================================

Start with the :doc:`Unitree G1 overview <g1/overview>` for the 50-task
benchmark, 2,000 paired demonstrations, and release artifacts. The
:doc:`G1 installation guide <g1/installation>` covers a new simulation setup.
Then use the :doc:`evaluation guide <g1/evaluation>` to connect an inference
service on port 8000 and run the benchmark.

The guides below cover simulation, datasets, and policy evaluation. See the
`upstream RoboCasa project <https://robocasa.ai>`_ and the repository README
for attribution, licensing, and paper citations.

.. raw:: html

   <style>
     /* Index-only: ensure toctree lists show bullet markers like the sample page. */
     .bd-article .toctree-wrapper ul {
       list-style-type: disc !important;
       padding-left: 1.4rem !important;
       margin-top: 0.4rem !important;
     }
     /* pydata/sphinx-book-theme often sets list-style: none on these; force it back. */
     .bd-article .toctree-wrapper ul > li {
       list-style-type: disc !important;
       display: list-item !important;
     }
     .bd-article .toctree-wrapper ul > li::marker {
       color: currentColor !important; /* ensure marker isn't "invisible" */
     }
     .bd-article .toctree-wrapper li {
       margin: 0.25rem 0 !important;
     }
   </style>

.. toctree::
   :maxdepth: 1
   :caption: Unitree G1

   g1/overview
   g1/installation
   g1/tasks
   g1/datasets
   g1/data_collection
   g1/dataset_processing
   g1/evaluation
   g1/reproducibility

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: introduction

   introduction/overview
   introduction/installation
   introduction/basic_usage
   introduction/codebase_overview

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: tasks

   tasks/atomic_tasks
   tasks/composite_tasks

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: datasets

   datasets/datasets_overview
   datasets/using_datasets
   .. datasets/human_datasets
   .. datasets/mimicgen_datasets

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: policy learning

   benchmarking/benchmarking_overview
   benchmarking/policy_learning_algorithms
   benchmarking/multitask_learning
   benchmarking/foundation_model_learning
   benchmarking/lifelong_learning

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: assets

   assets/scenes
   assets/objects
   assets/fixtures

.. toctree::
   :maxdepth: 1
   :caption: RoboCasa reference: use cases

   use_cases/creating_datasets
   .. use_cases/creating_tasks
   .. use_cases/mimicgen

.. toctree::
   :maxdepth: 4
   :caption: Source API

   api/robocasa

.. toctree::
   :maxdepth: 1
   :caption: Previous Versions

   v0.2 <v0_2>
 




Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
