.. RoboCasa documentation master file, created by
   sphinx-quickstart on Tue Jun 18 15:47:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RoboCasa's documentation!
===============================

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
   :caption: Introduction

   introduction/overview
   introduction/installation
   introduction/basic_usage
   introduction/codebase_overview

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   tasks/atomic_tasks
   tasks/composite_tasks

.. toctree::
   :maxdepth: 1
   :caption: Datasets

   datasets/datasets_overview
   datasets/using_datasets
   .. datasets/human_datasets
   .. datasets/mimicgen_datasets

.. toctree::
   :maxdepth: 1
   :caption: Benchmarking

   benchmarking/benchmarking_overview
   benchmarking/policy_learning_algorithms
   benchmarking/multitask_learning
   benchmarking/foundation_model_learning
   benchmarking/lifelong_learning

.. toctree::
   :maxdepth: 1
   :caption: Assets

   assets/scenes
   assets/objects
   assets/fixtures

.. toctree::
   :maxdepth: 1
   :caption: Use Cases

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
* :ref:`modindex`
* :ref:`search`
