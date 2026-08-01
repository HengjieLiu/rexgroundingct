# Experiment plans

Each child directory is an immutable, stable-ID experiment package. The
machine-readable plan is canonical, state is mutable only through
`experimentctl`, and heavyweight outputs live behind the ignored `runtime`
link.
