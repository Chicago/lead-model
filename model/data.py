import pandas as pd
from sklearn import preprocessing
from functools import partial
from scipy import stats
import re
import numpy as np
import collections

import random
import datetime
from lead.output.aggregate import aggregate
from lead.output import tests_aggregated,buildings_aggregated

from drain import util
from drain.util import prefix_columns, join_years
from drain.data import *

import warnings

CATEGORY_CLASSES = {
#    'address_building_condition': ['SOUND', 'NEEDS MINOR REPAIR',
#        'NEEDS MAJOR REPAIR', 'UNINHABITABLE'],
    'kid_sex' : ['M', 'F'],
    'test_type' : ['V', 'C'],
    'surname_ethnicity': ['black', 'white', 'api', 'aian', 'p2race'],
    'kid_ethnicity': ['black', 'white', 'hispanic', 'asian'],
    'tract_ethnicity': ['asian', 'black', 'white', 'latino'],
    'ward_ethnicity': ['black', 'white', 'latino'], # there are no majority asian wards
    'ward_id': range(1,51),
    'community_area_id': range(1,78)
}

class LeadData(ModelData):
    # default exclusions set
    # TODO: organize and explain these
    EXCLUDE = { 
        'kid_id', 'test_id', 'address_id', 'complex_id', 'census_tract_id', # ids
        'kid_first_name', 'kid_last_name', 'address_method', 'address', # strings
        'test_bll', 'test_minmax', 'kid_minmax_date', 'kid_max_bll', 'kid_max_date', # leakage
        'test_date', 'kid_date_of_birth', #date 
        'join_year', 'aggregation_end',# used for join
        'kid_birth_days_to_test',  'address_inspection_init_days_to_test' # variables that confuse the model?
    }

    DATE_COLUMNS = {
        'inspections' : ['init_date', 'comply_date']
    }
    
    KIDS_DATE_COLUMNS = ['kid_date_of_birth', 'test_date', 
                    'address_inspection_init_date', 'address_inspection_comply_date']
    
    def __init__(self, source, directory=None, 
		    tables=['inspections', 'addresses', 'acs']):
        self.source = source
        self.directory = directory
        
        self.tables = {t:None for t in tables}
    
    def read(self, **args):
        if self.source == 'pkl':
            self.read_pkl()
        elif self.source == 'sql':
            self.read_sql()
        elif self.source == 'csv':
            self.read_csv()
        else:
            raise ValueError('Unexpected data source: ' + str(self.source))
    
    def read_sql(self):
        engine = util.create_engine()
        
        self.tests = pd.read_sql('select * from output.tests', engine)
        
        for table in self.tables:
            date_columns = self.DATE_COLUMNS[table] if table in self.DATE_COLUMNS else None
            self.tables[table] = pd.read_sql('select * from output.' + table, engine, parse_dates=date_columns)
            
    def write(self):
        self.tests.to_pickle(self.directory + '/tests.pkl')
        
        for table in self.tables:
            self.tables[table].to_pickle(self.directory + "/" + table + '.pkl')
        
    def read_pkl(self):
        self.tests = pd.read_pickle(self.directory + "/tests.pkl")
        
        for table in self.tables:
            self.tables[table] = pd.read_pickle(self.directory + "/" + table + '.pkl')
    
    def read_csv(self):
        self.tests = pd.read_csv(self.directory + "/tests.csv", index_col='test_id')
        
        for c in self.KIDS_DATE_COLUMNS:
            self.tests[c] = self.tests[c].apply(lambda x: 
                    x if pd.isnull(x) else datetime.datetime.strptime(x, '%Y-%m-%d').date())
        
        for table in self.tables:
            self.tables[table] = pd.read_csv(self.directory + "/" + table + '.csv')
            
    def transform(self, year, train_years,
                kid, tract, address, ward, address_history, tract_history, # boolean true or false
                bll_threshold=5, require_address=True, 

                # these parameters have defaults that were established by testing
                spacetime_normalize_method = None, # whether or not to normalize each year of tract data
                test_aggregations={},
                building_aggregations={},
                max_age = None,
                min_age = None,
                training='all', # all, preminmax
                testing='all',  # all, never_tested
                training_max_test_age=None,
                testing_max_test_age=None,
                testing_max_today_age=None,
                community_area = False, # don't include community area binaries
                exclude={}, 
                undersample=None,
                impute=True, normalize=True, drop_collinear=False,
                ward_id = None, # filter to this particular ward
                building_year_decade=True,
                test_date_season=True,
                cluster_columns={}, # dict of ('column_name': n_clusters) pairs
                exclude_addresses=[296888, # Aunt Martha's Health Center
                                   70798,  # Union Health Service Inc. 
                                   447803]): # Former Maryville Hospital


        # get tests relevant to date
        today = datetime.date(year, 1, 1)

        df = pd.concat((tests_aggregated.censor_tests(self.tests, today),
                        self.tests[self.tests.test_date >= today]), copy=False)

        date_from = datetime.date(year - train_years, 1, 1)
        date_mask = (df.test_date >= date_from)
        df = df[date_mask]
        df = df[df.kid_date_of_birth.notnull()]
#        df.kid_date_of_birth.fillna( (df[test_date] < today).kid_date_of_birth.mean() )

        exclude = self.EXCLUDE.union(exclude)
        df = df.merge(self.tables['addresses'], on='address_id', how='left', copy=False)

        if min_age is not None:
            df = df[(df.test_kid_age_days >=  min_age)]
        if max_age is not None: 
            df = df[(df.test_kid_age_days <= max_age)]

        if ward_id is not None:
            df = df[df.ward_id==ward_id]
            
        if  require_address:
            df = df[df.address_id.notnull() & df.ward_id.notnull() & df.census_tract_id.notnull() & df.community_area_id.notnull()]
            exclude.update({'address_null', 'ward_null', 'census_tract_null', 'community_area_null'})
        if exclude_addresses is not None:
            df = df[~df.address_id.isin(exclude_addresses)]


        # cross validation
        df.set_index('test_id', inplace=True)
        train = (df.test_date < today) 
        if training == 'preminmax':
            train = train & (df.test_date <= df.kid_minmax_date)
        elif training == 'max':
            train = train & ( (df.test_bll > 5) == (df.kid_max_bll > 5) )

        if training_max_test_age is not None:
            train = train & (df.test_kid_age_days <= training_max_test_age)
      
        if testing == 'all':
            test = (df.test_date >= today) & (df.kid_date_of_birth < today) 
        elif testing == 'untested':
            test = (df.test_date >= today) & (df.kid_date_of_birth < today) & (df.min_sample_date >= today)
        else:
            print 'Warning: testing option \'{}\' not supported'.format(testing)

        if testing_max_test_age is not None:
            test = test & (df.test_kid_age_days <= testing_max_test_age)
        if testing_max_today_age is not None:
            test = test & (df.kid_date_of_birth.apply(lambda d: (today - d).days <= testing_max_today_age))

        # want to get a single test for each future kid
        # if they get poisoned, take their first poisoned test
        # if they don't, take their first test
        df2 = df[test]# & ((df.test_bll > 5) == (df.kid_minmax_bll > 5))]
        testix = df2.groupby('kid_id')['test_kid_age_days'].idxmin()
        first_test = pd.Series(df.index.isin(testix), index=df.index)
        test = first_test & ( ((df.kid_minmax_date >= today) & (df.kid_minmax_bll > 5)) | (df.kid_minmax_bll <= 5))

        train_or_test = train | test
        train = train.loc[train_or_test]
        test = test.loc[train_or_test]
        self.cv = (train,test)
        df = df[train_or_test]

        # set test details for (future) test set to nan to eliminate leakage!
        # can't know about minmax bll,date for future poisoningsa!
        # since test_id is currently the index it does not get cleared!
        test_columns = [c for c in df.columns if c.startswith('test_')]
        df.loc[ test, test_columns ] = np.nan

        df.reset_index(inplace=True)

        # generate a fake test date for future tests
        epoch = datetime.date.fromtimestamp(0)
        past_test = df.test_date < today # aka training
        mean_age = datetime.timedelta(df['test_kid_age_days'].mean())
        future_test_date = df['kid_date_of_birth'].apply(lambda d: max(d+mean_age, today))
        df['test_date'] = df['test_date'].where(past_test, future_test_date)

        df['kid_date_of_birth_month'] = df['kid_date_of_birth'].apply(lambda d: d.month)
        df['kid_birth_date'] = df['kid_date_of_birth']
    
        for c in ['kid_birth']: #['address_inspection_init', 'address_inspection_comply', 'kid_birth']:
            df[c + '_days'] = df[c + '_date'].apply(lambda d: None if pd.isnull(d) else (d - epoch).days)
            df[c + '_days_to_test'] = pd.to_timedelta((df['test_date'] - df[c + '_date']), 'D').astype(int)
            df.drop(c + '_date', axis=1, inplace=True)
        
        df['join_year'] = df.test_date.apply(lambda d: min(d.year-1, year-1)) 
        df['aggregation_end'] = df.test_date.apply(lambda d: util.datetime64(min(d.year, year), today.month, today.day)) 
        
        if test_date_season:
            df['test_date_month'] = df['test_date'].apply(lambda d: d.month).where(past_test)
            CATEGORY_CLASSES['test_date_month'] = range(1,13)

        # spatial 
        if not kid:
            exclude.add('kid_.*')
        if not address:
            #TODO update this for complexes
            exclude.update(['address_building_.*', 'address_assessor_.*', 'address_lat', 'address_lng'])
        if not ward:
            exclude.add('ward_id')
        if not community_area:
            exclude.add('community_area_id')
        
        # spatio-temporal
        years = range(year-train_years, year)
        end_dates = df['aggregation_end'].unique()
        engine = util.create_engine()
        
        all_levels = ['address_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id']
        left = df[ all_levels + ['join_year', 'aggregation_end']].drop_duplicates()
        spacetime = get_aggregation('output.tests_aggregated', test_aggregations, engine, end_dates=end_dates, left=left, prefix='tests')
        space = get_building_aggregation(building_aggregations, engine, left=left)
        
        inspections_tract_ag,inspections_address_ag = self.aggregate_inspections(years, levels=['census_tract_id', 'complex_id'])
        prefix_columns(inspections_tract_ag, 'tract_inspections_all_')
        prefix_columns(inspections_address_ag, 'address_inspections_all_')
        
        spacetime = spacetime.merge(inspections_tract_ag, how='left', left_on=['census_tract_id', 'join_year'], right_index=True, copy=False)
        spacetime = spacetime.merge(inspections_address_ag, how='left', left_on=['complex_id', 'join_year'], right_index=True, copy=False)

        # acs data
        left = df[['census_tract_id', 'join_year']].drop_duplicates()
        acs = self.tables['acs'].set_index(['census_tract_id', 'year'])
        prefix_columns(acs, 'acs_5yr_')
        # use outer join for backfilling
        acs = left.merge(acs, how='outer', left_on=['census_tract_id', 'join_year'], right_index=True, copy=False)
        acs_filled = acs.groupby('census_tract_id').transform(lambda d: d.sort('join_year').fillna(method='backfill'))
        # left join and groupby preserved the left index but grupby dropped the tract
        # so put the tract back
        acs_filled['census_tract_id'] = acs['census_tract_id']

        spacetime = spacetime.merge(acs_filled, on=['census_tract_id', 'join_year'], how='left', copy=False)
        spacetime.set_index(['address_id', 'aggregation_end'], inplace=True)
        spacetime.drop(['census_tract_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id', 'join_year'], axis=1, inplace=True)
        spacetime.fillna(0, inplace=True)

        if spacetime_normalize_method is not None:
            spacetime = spacetime.groupby(level='aggregation_end').apply(lambda x: util.normalize(x, method=spacetime_normalize_method))

        df = df.merge(spacetime, left_on=['address_id', 'aggregation_end'], right_index=True, how='left', copy=False )

        space.set_index(['address_id', 'aggregation_end'], inplace=True)
        space.drop(['census_tract_id', 'building_id', 'complex_id', 'census_block_id', 'census_tract_id', 'ward_id', 'community_area_id', 'join_year'], axis=1, inplace=True)
        df = df.merge(space, left_on=['address_id', 'aggregation_end'], right_index=True, how='left', copy=False )

        if not address_history:
            exclude.update(['address_inspections_.*', 'address_tests_.*'])
        if not tract_history:
            exclude.update(['tract_inspections_.*', 'tract_tests_.*', 'acs_5yr_.*'])
        
        #if building_year_decade:
            #df['complex_building_year_decade'] = (df['complex_building_year'] // 10)
            #CATEGORY_CLASSES['complex_building_year_decade'] =  df['complex_building_year_decade'].dropna().unique()


        df.set_index('test_id', inplace=True)

        for column, n_clusters in cluster_columns.iteritems():
            binarize_clusters(df, column, n_clusters, train=train)

        X,y = Xy(df, y_column = 'kid_minmax_bll', exclude=exclude, impute=impute, normalize=normalize, train=train, category_classes=CATEGORY_CLASSES)
 
        self.X = X
        self.y = y > bll_threshold

        if drop_collinear:
            util.drop_collinear(X)
        
        if undersample is not None:
            # undersample is the desired *proportion* of the majority class
            # calculate p, the desired proportion by which to undersample
            y_train = self.y[self.cv[0]]
            T = y_train.sum()
            F = len(y_train) - T
            p = (undersample)*T/((1-undersample)*F)
            self.cv = (undersample_cv(df, self.cv[0], p), self.cv[1])

    def aggregate_inspections(self, years, levels):
        inspections = self.tables['inspections']
        #res_columns = {'res_count': {'numerator': 'residential'}}
        #for index in levels:
        #    count = aggregate(self.tables['addresses'], res_columns, index=index)
        #    count.columns = [index + '_res_count']
        #    inspections = inspections.merge(count.reset_index(), how='left', on=index)
        
        for column in ['hazard_int','hazard_ext']:
            inspections[column].fillna(True, inplace=True)
            inspections[column] = inspections[column].astype(int)
        
        inspections['year'] = inspections['init_date'].fillna(inspections['comply_date']).apply(lambda d:d.year)
        inspections = join_years(inspections, years)
        
        comply_not_null = inspections[inspections.comply_date.notnull()]
        inspections['comply'] = (comply_not_null['comply_date'].apply(lambda d: d.year) < comply_not_null.year)
        inspections['comply'] = inspections['comply'].fillna(False).astype('int')
        dt = (inspections['comply_date'] - inspections['init_date']).where(inspections['comply'])
        inspections['days_to_compliance'] = dt[dt.notnull()] / np.timedelta64(1, 'D')
        
        INSPECTION_COLUMNS = {
            'count': {'numerator':1},
            'inspected': {'numerator':1, 'func': np.max},
            'hazard_int_count': {'numerator':lambda i: i['hazard_int'] & ~i['comply']},
            'hazard_ext_count': {'numerator':lambda i: i['hazard_ext']},
            'hazard_int_prop': {'numerator':'hazard_int', 'denominator':1},
            'hazard_ext_prop': {'numerator':'hazard_ext', 'denominator':1},
            'compliance_count': {'numerator': 'comply'},
            'compliance_prop': {'numerator': 'comply', 'denominator': 1},
            'avg_init_to_comply_days': {'numerator': 'days_to_compliance', 'func':'mean'},
        }
        
        r = []
        for level in levels:
            #INSPECTION_COLUMNS['pct_inspected'] = {'numerator': 1, 'denominator': level + '_res_count', 'denominator_func': np.max}
            r.append(aggregate(inspections, INSPECTION_COLUMNS, index=[level,'year']))
            
        return r
    
def get_building_aggregation(building_aggregations, engine, left=None):
    df = get_aggregation('output.buildings_aggregated', building_aggregations, engine, left=left)
    
    not_null_columns = [c for c in df.columns if c.endswith('_not_null')]
    df.loc[:,not_null_columns].fillna(False, inplace=True)

    return df

# left is an optional dataframe with index <level>, aggregaton_end
# it is left-joined to ensure returned df has the specified rows
