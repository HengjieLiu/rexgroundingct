# ASD-030 required tests

The implementation stage must add tests for:

- known-positive, certified-negative, and unknown precedence;
- 15-mm physical dilation and orientation invariance;
- certification rejection after anatomy QC failure;
- zero negative gradient on unknown voxels;
- deterministic candidate ranks and replay hashes;
- count-matched controls and patient separation;
- val120/val200 denial before promotion;
- epoch-zero matched checkpoint equivalence;
- atomic completion markers and full-state resume.
