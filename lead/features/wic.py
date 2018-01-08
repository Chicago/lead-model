import os
import logging

import pandas as pd
import numpy as np

from drain import util, aggregate, data
from drain.aggregate import Aggregate, Count, aggregate_counts, days
from drain.aggregation import SpacetimeAggregation
from drain.step import Construct
from drain.data import FromSQL, binarize, binarize_sets, select_regexes
from drain.util import list_filter_none, union

enroll = FromSQL(query="""
        with enroll as (
        SELECT kid_id, p.* 
        FROM cornerstone.partenrl p join aux.kid_wics using (part_id_i)
        UNION ALL
        SELECT kid_id, p.*
        FROM cornerstone.partenrl p join aux.kid_mothers on p.part_id_i = mothr_id_i)

select kid_id, register_d, last_upd_d,
    med_risk_f = 'Y' as medical_risk,
    clinicid_i as clinic,
    emplymnt_c as employment, 
    occptn_c as occupation,
    hsehld_n as household_size, hse_inc_a / 100000.0 as household_income,
array_remove(array[lang_1_c, lang_2_c, lang_3_c], null) as language,
array_remove(array[pa_cde1_c, pa_cde2_c, pa_cde3_c, pa_cde4_c, pa_cde5_c], null) as assistance
from enroll 
""", tables=['aux.kid_wics', 'aux.kid_mothers'], parse_dates=['register_d', 'last_upd_d'])

enroll2 = Construct(binarize, inputs=[enroll], category_classes=['employment', 'occupation', 'clinic'], min_freq=100)

enroll3 = Construct(binarize_sets, inputs=[enroll2], columns=['assistance', 'language'], cast=True, min_freq=100)
enroll3.target=True

class EnrollAggregation(SpacetimeAggregation):
    def __init__(self, spacedeltas, dates, parallel=False):
        SpacetimeAggregation.__init__(self,
                inputs = [enroll3],
                spacedeltas = spacedeltas,
                dates = dates,
                prefix = 'wicenroll',
                date_column = 'register_d', 
                parallel=parallel)

    def get_aggregates(self, date, delta):
        enroll = self.inputs[0].result
        aggregates = [
            Aggregate('medical_risk', 'any', fname=False),
            Aggregate(['household_size', 'household_income'], 
                      ['median', 'max']),
            Aggregate(list(select_regexes(enroll, ['(employment|occupation|language|assistance|clinic)_.*'])), 'sum', fname=False)
        ]

        return aggregates


births = FromSQL(query="""
        SELECT *, 
        apgar_n::int as apgar,
        nullif(lgt_inch_n, 0) as length,
        nullif(wgt_grm_n, 0) as weight,
        nullif(headcirc_n, 0) as head_circumference,
        array_remove(array[
                    inf_cmp1_c, inf_cmp2_c, inf_cmp3_c, inf_cmp4_c, inf_cmp5_c
                    ], null) as complication,
                    brth_typ_c as place_type,
                    inf_disp_c as disposition
        FROM aux.kids
        JOIN aux.kid_mothers USING (kid_id)
        JOIN cornerstone.birth USING (part_id_i, mothr_id_i)
        """, tables=['aux.kids', 'aux.kid_mothers'], parse_dates=['date_of_birth'])


births2 = Construct(binarize, inputs=[births], category_classes=['place_type', 'disposition'])
births3 = Construct(binarize_sets, inputs=[births2], columns=['complication'], cast=True)
births3.target = True

class BirthAggregation(SpacetimeAggregation):
    def __init__(self, spacedeltas, dates, parallel=False):
        SpacetimeAggregation.__init__(self,
                inputs = [births3],
                spacedeltas = spacedeltas,
                dates = dates,
                prefix = 'wicbirth',
                date_column = 'date_of_birth', 
                parallel=parallel)

    def get_aggregates(self, date, delta):
        births = self.inputs[0].result
        aggregates = [
            Aggregate('length', 'max', fname=False),
            Aggregate('weight', 'max', fname=False),
            Aggregate('head_circumference', 'max', fname=False),
            Aggregate('apgar', 'max', 'apgar_score', fname=False),
            Aggregate(list(select_regexes(births, ['(complication|place_type|disposition)_.*'])), 'sum', fname=False),
            Aggregate(lambda b: b.apors_f == 'Y', 'any', 'apors', fname=False),
            Aggregate(lambda b: b.icu_f == 'Y', 'any', 'icu', fname=False),
        ]

        return aggregates

prenatal = FromSQL("""
SELECT kid_id, date_of_birth, p.*, serv_typ_c as service
FROM aux.kids
JOIN aux.kid_mothers USING (kid_id)
JOIN cornerstone.birth b USING (part_id_i, mothr_id_i)
JOIN cornerstone.prenatl p ON b.mothr_id_i = p.part_id_i
where date_of_birth - visit_d between -365 and 365
""", tables=['aux.kids', 'aux.kid_mothers'], parse_dates=['date_of_birth', 'visit_d'])
prenatal.target = True

prenatal2 = Construct(binarize, inputs=[prenatal], category_classes=['service'])

class PrenatalAggregation(SpacetimeAggregation):
    def __init__(self, spacedeltas, dates, parallel=False):
        SpacetimeAggregation.__init__(self,
                inputs = [prenatal2],
                spacedeltas = spacedeltas,
                dates = dates,
                prefix = 'wicprenatal',
                date_column = 'visit_d', 
                parallel=parallel)

    def get_aggregates(self, date, delta):
        prenatal = self.inputs[0].result

        aggregates = [
            Count(),
            Aggregate(days('visit_d', 'date_of_birth'), ['min', 'max'], 'visit'),
            Aggregate(list(select_regexes(prenatal, ['service_.*'])), 'sum', fname=False),
            Aggregate('preg_nbr_n', 'max', 'previous_pregnancies', fname=False),
            Aggregate('lv_brth_n', 'max', 'previous_births', fname=False),
            Aggregate('othr_trm_n', 'max', 'previous_terminations', fname=False),
            Aggregate(lambda p: p.smk3_mth_f == 'Y', 'any', 'smoked_3mo', fname=False),
            Aggregate('cig3_day_n', 'max', 'cigarettes_per_day', fname=False),
            Aggregate(lambda p: p.drk3_mth_f == 'Y', 'any', 'drank_3mo', fname=False),
            Aggregate('dr_dy_wk_n', 'max', 'days_drank_per_week', fname=False),
            Aggregate('drnk_day_n', 'max', 'drinks_per_day', fname=False),
        ]

        return aggregates
