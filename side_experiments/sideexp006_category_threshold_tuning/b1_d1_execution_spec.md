# B1 and D1 execution contract

Authorized scope: fixed val200, 200 cases / 381 findings; thresholds 0.05–0.95
inclusive, raw finding Dice and Dice >= 0.1 hit rate. No adoption, instance
metrics, E1, test inference, or submission changes. Preserve completed A1.

B1 freezes the nine-model registry routing, reuses four strict full caches,
and exports only 81 missing routed findings (69 model–case evaluations):
Exp007 e025 1/1; Exp011 e100 6/6; Exp012 22/19; Exp013 49/40;
Exp017 absolute e125 3/3. Exp011 uses shared train/val native linear-HU.
Subset arrays have original finding IDs and explicit channel maps. Established
predictor, padding, crop restoration and GT orientation functions are reused.
Float16 clipped [-30,30] is accepted only with exact same-pass zero-mask parity;
otherwise keep float32. Hash checkpoint, plans, preprocessing manifest, inputs,
GT, prompts, geometry, code, and logical array values excluding NPY headers.
Subset caches must never be advertised as complete val200 caches.

One GPU at a time, one checkpoint load, wait for >=20 GiB free; retain the
largest cached-volume case as smoke before the rest. Existing jobs remain
untouched. D1 runs independently with two single-thread CPU workers, followed
by B1 with two workers. Dispatch remains bounded; source drift rejects resume.
D1 sums float32 sigmoid probabilities in its frozen four-model order and tests
sum >= 4*t. B1 reads only its routed sources and channels.

D1 0.50 gate: Dice 0.3575041942161927 within 1e-10 and 296 hits.
B1 gate is independently composed from source per-finding same-pass references;
require per-finding and overall Dice agreement within 1e-10 and exact hits.
Historical B1 0.3602780325749142 / 298 is displayed separately.

Each run must contain 7239 unique rows, all 14 categories including empty 2f,
category recomposition, comparison CSV, manifests, progress, executed notebook,
embedded figures, HTML and verification. Maxima tie by proximity to 0.50, then
lower threshold. Small categories n<10 are labelled. Source code and focused
tests must pass before launch; run repository checks and visually review plots.
