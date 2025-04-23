"""
Data class representing the blast radius information for a table.
"""

import dataclasses


@dataclasses.dataclass
class BlastRadiusInfo:
    # An overall indication of the blast radius severity, integer [1,4]. Higher -> more severe.
    impact_level: int
    # The number of immediately downstream tables
    num_downstream_tables: int
    # The total number of queries over the last 30 days, across the current and downstream tables
    num_queries_on_affected_tables: int
