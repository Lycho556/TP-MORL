# P8-C2: delineation with the same cell-level future (written before running, 2026-11-09)

Question. In P8-C each delineation drew its own unit-level fields, so a field draw of the 8-, 12-
and 18-cell inventories is not the same future (correlation of the value of information across
delineations for the same draw: 0.20). Holding the future fixed, does the delineation itself change
the balance between information and coordination?

Design. opportunity.CELL_FIELD: the fields are drawn once on the 7,553 candidate cells of the
100 m grid (same generator, same field seed, cells in row-major order, infrastructure clusters on
cell coordinates) and assigned to units:
  * rep  (primary): each unit takes the field of its representative cell, the member cell nearest
    the unit centroid; the distribution of unit fields is the same as in the paper.
  * mean (secondary): each unit takes the mean field over its cells (categorical arrays and onsets
    from the representative cell); this also smooths the fields of larger units.
Targets 8, 12, 18 cells; fields: main field seed and scenario draws 1-10; listing draws 7, 0, 1, 2,
3, 4; M = 20; rule-based rules only. info = trajectory - persistence, coord = rh_full - trajectory,
per field averaged over listing draws.

Primary measure (rep): paired over the 11 fields, D18 = (coord - info) at 18 cells minus that at
12 cells, and D8 likewise; mean and 95% t-interval; and the count of fields with coord > info at
each target.
  * Delineation effect present: the interval of D18 (or D8) excludes 0.
  * Reversal caused by delineation: coord > info in at least 9 of 11 fields at 18 cells.
  * Reversal not caused by delineation: coord > info in at most 5 of 11 fields at 18 cells.
The mean mode is reported whatever the outcome.
