"""
ATSPM Reports Package (Imperative Shell)

Orchestrates data fetching, plot generation, and HTML output.
No analysis logic lives here — this package calls the functional core
(src/atspm/analysis/) and plotting (src/atspm/plotting/) via the data
reader (src/atspm/data/reader.py).

Modules:
    generators: PlotGenerator class and generate_reports() convenience
                function for producing HTML report files per intersection
                per date.
"""

from .generators import (
    PlotGenerator,
    generate_reports,
)

__all__ = [
    'PlotGenerator',
    'generate_reports',
]
