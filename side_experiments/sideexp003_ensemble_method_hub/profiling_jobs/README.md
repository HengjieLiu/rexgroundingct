# Isolated pipeline profiling jobs

Bounded timing experiments, separate from production cache jobs. Each job freezes
its inputs and source before launch; runtime arrays remain on shared/local data
mounts. These jobs never resume production waves.

- [p001_test300_pipeline16_gpu8](p001_test300_pipeline16_gpu8/README.md): one GPU
  pass for 16 model–case pairs and CPU replays at 4/8/16 workers.
