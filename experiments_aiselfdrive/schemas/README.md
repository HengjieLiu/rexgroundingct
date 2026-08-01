# Self-drive schemas

The JSON Schemas in this directory are the authoritative structural contracts
for portfolio, experiment, state, event, claims, artifacts, and results.
`experimentctl validate --all` validates every machine-readable file against
them and performs semantic checks that JSON Schema cannot express.
