from drain.step import Step
from drain.util import timestamp, cross_join
from drain.data import FromSQL, Merge

import pandas as pd
import numpy as np
import logging

# in addition to all addresses, we add all cells in the partition
# created by intersecting blocks, wards and communities 
# in anticipation of any new addresses in deployment
addresses = FromSQL("""
with blocks as (
select
    b.geoid10::double precision census_block_id,
    substring(b.geoid10 for 11)::double precision census_tract_id,
    c.area_numbe::int community_area_id,
    w.ward::int ward_id
from input.census_blocks b
join input.community_areas c
    on st_intersects(b.geom, c.geom)
join input.wards w
    on st_intersects(b.geom, w.geom) and st_intersects(c.geom, w.geom)
group by 1,2,3,4
)
select
    null address,
    null address_lat,
    null address_lng,
    null as address_id,
    null as building_id,
    null as complex_id, *
from blocks
UNION ALL
select address, address_lat, address_lng, 
    address_id, building_id, complex_id,
    census_block_id, census_tract_id, 
    community_area_id, ward_id
from output.addresses
    """, tables=['output.addresses', 'input.census_blocks', 'input.census_tracts', 'input.community_areas', 'input.wards'])
addresses.target = True

class LeadAddressLeft(Step):
    """
    This Step simply adds dates to all addresses in the database. It is used
    by LeadData for building an address dataset.
    """
    def __init__(self, month, day, year_min, year_max):
        """
        Args:
            month: the month to use
            day: the day of the month to use
            year_min: the year to start
            year_max: the year to end
        """
        Step.__init__(self, month=month, day=day, year_min=year_min, year_max=year_max, inputs=[addresses])

    def run(self, addresses):
        """
        Returns:
            - left: the cross product of the output.addresses table with the
                specified dates.
        """
        dates = [timestamp(year, self.month, self.day)
                 for year in range(self.year_min, self.year_max+1)]
        if len(dates) == 1:
            # when there's exactly one date modify in place for efficiency
            addresses['date'] = dates[0]
            left = addresses
        else:
            left = cross_join(addresses, pd.DataFrame(dates))
            
        return {'left':left}
