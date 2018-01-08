from drain import data, step, model, data
from drain.util import dict_product
from drain.step import Call, Construct, MapResults

from itertools import product
import pandas as pd
import os

import lead.model.data
import lead.model.transform
import lead.model.cv
from lead.features import aggregations


def bll6_forest():
    """
    The basic temporal cross-validation workflow
    """
    return bll6_models(forest())


def bll6_forest_today():
    """
    The workflow used to construct a current model
    Parses the environment variable TODAY using pd.Timestamp to set the date
    """
    today = pd.Timestamp(os.environ['TODAY'])
    p = bll6_models(
            forest(),
            dict(year=today.year,
                 month=today.month,
                 day=today.day))[0]
    # save the model
    p.get_input('fit').target = True
    p.get_input('mean').target = True

    # put the predictions into the database
    tosql = data.ToSQL(table_name='predictions', if_exists='replace',
            inputs=[MapResults([p], mapping=[{'y':'df', 'feature_importances':None}])])
    tosql.target = True
    return tosql

def address_data_past():
    """
    Builds address-level features for the past
    Plus saves fitted models and means for the past
    """
    ds = [] # lead address data
    for y in range(2011,2014+1):
        d = lead.model.data.LeadData(
                year_min=y,
                year_max=y,
                month=1,
                day=1,
                address=True)
        d.target = True
        ds.append(d)

    ps = bll6_forest() # predictions
    for p in ps:
        p.get_input('fit').target = True
        p.get_input('mean').target = True

    return ds + ps
 
def address_data_today():
    """
    Builds address-level features today
    """
    today = pd.Timestamp(os.environ['TODAY'])
    d = lead.model.data.LeadData(
            year_min=today.year,
            year_max=today.year,
            month=today.month,
            day=today.day,
            address=True)
    d.target = True

    return d
    
def bll6_forest_quick():
    """
    A fast lead model that only uses 1 year of training data
    """
    today = pd.Timestamp(os.environ['TODAY'])
    p = bll6_models(
            forest(n_estimators=10),
            dict(year=today.year,
                 month=today.month,
                 day=today.day,
                 train_years=1))[0]
    return p


def bll6_forest_quarterly():
    """
    Quarterly forest models
    """
    return bll6_models(forest(),
        {'month':[1,4,7,10], 'year':range(2010,2014+1)})

def bll6_forest_monthly():
    """
    Monthly forest models
    """
    return bll6_models(forest(),
        {'month':range(1,13), 'year':range(2010,2014+1)})

def forest(**update_kwargs):
    """
    Returns a step constructing a scikit-learn RandomForestClassifier
    """
    kwargs = dict(
        _class='sklearn.ensemble.RandomForestClassifier',
        n_estimators=1000,
        n_jobs=int(os.environ.get('N_JOBS', -1)),
        criterion='entropy',
        class_weight='balanced_bootstrap',
        max_features='sqrt',
        random_state=0)

    kwargs.update(**update_kwargs)

    return [step.Construct(**kwargs)]

def bll6_models(estimators, cv_search={}, transform_search={}):
    """
    Provides good defaults for transform_search to models()
    Args:
        estimators: list of estimators as accepted by models()
        transform_search: optional LeadTransform arguments to override the defaults

    """
    cvd = dict(
        year=range(2011, 2014+1),
        month=1,
        day=1,
        train_years=[6],
        train_query=[None],
    )
    cvd.update(cv_search)

    transformd = dict(
        wic_sample_weight=[0],
        aggregations=aggregations.args,
        outcome_expr='max_bll0 >= 6',
        outcome_where_expr='max_bll0 == max_bll0' # this means max_bll0.notnull()
    )
    transformd.update(transform_search)
    return models(estimators, cvd, transformd)

def models(estimators, cv_search, transform_search):
    """
    Grid search prediction workflows. Used by bll6_models, test_models, and product_models.
    Args:
        estimators: collection of steps, each of which constructs an estimator
        cv_search: dictionary of arguments to LeadCrossValidate to search over
        transform_search: dictionary of arguments to LeadTransform to search over

    Returns: a list drain.model.Predict steps constructed by taking the product of
        the estimators with the the result of drain.util.dict_product on each of
        cv_search and transform_search.

        Each Predict step contains the following in its inputs graph:
            - lead.model.cv.LeadCrossValidate
            - lead.model.transform.LeadTransform
            - drain.model.Fit
    """
    steps = []
    for cv_args, transform_args, estimator in product(
            dict_product(cv_search), dict_product(transform_search), estimators):

        cv = lead.model.cv.LeadCrossValidate(**cv_args)
        cv.name = 'cv'

        X_train = Call('__getitem__', inputs=[MapResults([cv], {'X':'obj', 'train':'key',
                                                       'test':None, 'aux':None})])
        mean = Call('mean', inputs=[X_train])
        mean.name = 'mean'

        X_impute = Construct(data.impute,
                             inputs=[MapResults([cv], {'aux':None, 'test':None, 'train':None}),
                              MapResults([mean], 'value')])

        cv_imputed = MapResults([X_impute, cv], ['X', {'X':None}])
        cv_imputed.target = True

        transform = lead.model.transform.LeadTransform(inputs=[cv_imputed], **transform_args)
        transform.name = 'transform'

        fit = model.Fit(inputs=[estimator, transform], return_estimator=True)
        fit.name = 'fit'

        y = model.Predict(inputs=[fit, transform],
                return_feature_importances=True)
        y.name = 'predict'
        y.target = True

        steps.append(y)

    return steps
