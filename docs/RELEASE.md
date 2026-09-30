# Full implementation release

The initial core preview is replaced by the complete paper implementation. Public `shared_onset` imports delegate to the same `tri` implementation used by the research pipeline, including the original exception-base alias.

Release checks:

- Full suite: 1,294 tests passed in the research environment.
- Fresh Python 3.11 environment, without inherited packages: 736 inference/domain/public-API tests and 100 subtests passed; dependency checks passed.
- Fresh-environment smoke: standard product chain, Boundary, and Visible + Early each returned two verified samples on the same learned R=3 request.
- Base and motion-weighted examples exported verified completions; a released real request also exported playable MIDI.
- Aligned data preparation reproduced all arrays of the original 3,321-window dataset and chord sidecar exactly. Both the 24-request and 8-request selections matched the original cohorts exactly.
- Released inference checkpoint tensors equal the original best checkpoint. All 100 archived probability files are unchanged. Re-evaluating the neural network on a different numerical backend can give slightly different q; the supplied frozen tables define the exact inference comparisons.
- All 920 recorded rows were unpacked and summarized, and the paper's experiment figures rebuilt from archived observations. Full 920-row timing studies were not rerun during release preparation.
- Plans for all four released benchmark stages retain their original 480/168/144/128 row counts.

The original solver and sampler mathematics are unchanged. Release-specific edits provide portable paths, manifest-relative loading, and an explicit data/checkpoint entry point for fresh request preparation. Training defaults and original research modules are retained. New runs write to separate output directories.
