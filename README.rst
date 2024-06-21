.. class:: center

    :Title: Adversarial Schrödinger bridges on domain translation problem
    :Type: Master's Thesis
    :Author: Ksenofontov Gregory
    :Supervisor: Candidate of Physico-Mathematical Sciences, Isachenko Roman

Abstract
========

The master's thesis examines the domain translation problem. This problem is to find a mapping $G$ and $K$ that translate elements from the set $X$ to $Y$ and vice versa. The advanced method for solving this problem is Schrödinger bridges - direct and inverse stochastic processes that are limited by given distributions. Their main feature among other domain translation methods, such as CycleGAN, is the additional property of the optimality of the resulting translation.

The goal of the study is to develop a new method for finding Schrödinger bridges that adresses two problems: the need to model a stochastic process, as well as the curse of dimensionality. To solve these problems, it is proposed a new approach based on adversarial learning. This method combines the advantages of adversarial generative networks (GANs) and Schrödinger bridges. In this work, experiments of the proposed approach are carried out on various datasets, including 2D data and EMNIST. The results obtained show that the proposed method satisfies all the stated requirements.


Example
=======

.. figure:: thesis/images/2d_results.png
    :scale: 70%
    :align: center
    :alt: Adversarial Schrödinger bridges on 2D data
    
    Adversarial Schrödinger bridges on 2D data

.. Research publications
.. ===============================
.. 1. 

.. Presentations at conferences on the topic of research
.. ================================================
.. 1. 

Software modules developed as part of the study
======================================================
1. A python package *adversarial-sb* with all implementation `here <https://github.com/gregkseno/masters-thesis/tree/master/src>`_.
2. Notebooks with all experiment visualisation `here <https://github.com/gregkseno/masters-thesis/tree/master/experiments>`_.
