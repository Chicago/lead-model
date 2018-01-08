from drain.step import Step
from drain import util, data

import pandas as pd
import numpy as np
import logging

from lead.model.data import LeadData


class LeadTransform(Step):
    """
    This Step transforms the data for modeling by defining an outcome variable,
    performing feature selection and creating sample weights.
    """
    def __init__(self, inputs, outcome_expr, aggregations,
            outcome_where_expr=None, wic_sample_weight=0,
            exclude=[], include=[]):
        """
        Args:
            inputs: list containing a LeadCrossValidate step
            outcome_expr: the query to perform on the auxillary information to produce an outcome variable
            aggregations: defines which of the SpacetimeAggregations to include
                and which to drop
            outcome_where_expr: where to evaluate the outcome_expr,
                defaults to None, which means everywhere
            wic_sample_weight: optional different sample weight for wic kids
        """
        Step.__init__(self,
                inputs=inputs,
                outcome_expr=outcome_expr,
                aggregations=aggregations,
                outcome_where_expr=outcome_where_expr,
                wic_sample_weight=wic_sample_weight, 
                exclude=exclude, include=include)

    def run(self, X, aux, train, test):
        """
        Args:
            X: full feature matrix, including both training and test sets
            aux: auxillary information, aligned with X
            train: boolean series to mask training, aligned with X
            test: boolean series to mask testing, aligned with X

        """
        y = aux.eval(self.outcome_expr)
        if self.outcome_where_expr is not None:
            y = y.where(aux.eval(self.outcome_where_expr))

        logging.info('Selecting aggregations')
        aggregations = self.get_input(LeadData).aggregations
        for a, args in self.aggregations.iteritems():
            X = aggregations[a].select(X, args, inplace=True)

        logging.info('Selecting features')
        X = data.select_features(X, exclude=self.exclude, 
                include=self.include)

        sample_weight = 1 + (aux.wic * self.wic_sample_weight)

        c = data.non_numeric_columns(X)
        if len(c) > 0:
            logging.warning('Non-numeric columns: %s' % c)

        return {'X': X, 'y': y, 
                'train': train, 'test': test, 
                'aux': aux, 'sample_weight': sample_weight}
