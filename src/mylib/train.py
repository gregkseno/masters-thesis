#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The :mod:`mylib.train` contains classes:

- :class:`mylib.train.Trainer`

The :mod:`mylib.train` contains functions:

- :func:`mylib.train.cv_parameters`
'''
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SyntheticBernuliDataset(object):
    r'''Base class for synthetic dataset.'''
    def __init__(self, n=10, m=100, seed=42):
        r'''Constructor method

        :param n: the number of feature
        :type n: int
        :param m: the number of object
        :type m: int
        :param seed: seed for random state.
        :type seed: int
        '''
        rs = numpy.random.RandomState(seed)

        self.w = rs.randn(n) # Генерим вектор параметров из нормального распределения
        self.X = rs.randn(m, n) # Генерим вектора признаков из нормального распределения

        self.y = rs.binomial(1, expit(self.X@self.w)) # Гипотеза порождения данных - целевая переменная из схемы Бернули


class Trainer(object):
    r'''Base class for all trainer.'''
    def __init__(self, model, X, Y, seed=42):
        r'''Constructor method

        :param model: The class with fit and predict methods.
        :type model: object

        :param X: The array of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: numpy.array
        :param Y: The array of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: numpy.array

        :param seed: Seed for random state.
        :type seed: int
        '''
        self.model = model
        self.seed = seed
        (
            self.X_train, 
            self.X_val, 
            self.Y_train, 
            self.Y_val
        ) = train_test_split(X, Y, random_state=self.seed)

    def train(self):
        r''' Train model
        '''
        self.model.fit(self.X_train, self.Y_train)

    def eval(self, output_dict=False):
        r'''Evaluate model for initial validadtion dataset.
        '''
        return classification_report(
            self.Y_val, 
            self.model.predict(
                self.X_val), output_dict=output_dict)

    def test(self, X, Y, output_dict=False):
        r"""Evaluate model for given dataset.
        
        :param X: The array of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: numpy.array
        :param Y: The array of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: numpy.array
        """
        return classification_report(
            Y, self.model.predict(X), output_dict=output_dict)


def cv_parameters(X, Y, seed=42, minimal=0.1, maximum=25, count=100):
    r'''Function for the experiment with different regularisation parameters 
        and return accuracy and weidth for LogisticRegression for each parameter.

    :param X: The array of shape 
        `num_elements` :math:`\times` `num_feature`.
    :type X: numpy.array
    :param Y: The array of shape 
        `num_elements` :math:`\times` `num_answers`.
    :type Y: numpy.array

    :param seed: Seed for random state.
    :type seed: int
    :param minimal: Minimum value for the Cs linspace.
    :type minimal: int
    :param maximum: Maximum value for the Cs linspace.
    :type maximum: int
    :param count: Number of the Cs points.
    :type count: int
    '''

    Cs = numpy.linspace(minimal, maximum, count)
    parameters = []
    accuracy = []
    for C in Cs:
        trainer = Trainer(
            LogisticRegression(penalty='l1', solver='saga', C=1/C),
            X, Y,
        )

        trainer.train()

        accuracy.append(trainer.eval(output_dict=True)['accuracy'])
        
        parameters.extend(trainer.model.coef_)

    return Cs, accuracy, parameters
