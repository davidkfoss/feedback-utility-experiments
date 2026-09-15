# Release validation

Validated locally on 2026-09-15 before publication:

- All 205 input files match the recorded source SHA-256 hashes; the 60 RunPod controls are included. The archive contains exactly 205 flat JSON entries and is 41,989,056 bytes.
- A fresh archive extraction regenerated all two figures and nine tables using the isolated analysis-only environment. The URL-reading path was exercised independently with the same pinned ZIP.
- All displayed table numbers match the uploaded paper except the two documented appearances of the AG News uniform-random drop rounding discrepancy.
- Full-precision paired effects and bootstrap intervals match the original scripts within 1e-12. Figure 1 mean and SD arrays match within 1e-14.
- All 30 AEES donor traces and all 60 reconstructed control schedules pass the audit, including leave-one-out exclusion, Hamilton counts, action order, and episode boundaries.
- The full test suite passed: **25 tests** using the available training libraries. The smaller analysis-only environment passed **17 tests**, with two training-dependent test modules skipped.
- Every manifest training command parses to its recorded active configuration. The documented inactive baseline action-list normalization is explicit in the test.
- Five-step CPU checks match the original NLP function outputs for four policy cases, including parameter updates, RNG states, and final partial episodes. All 27 extracted shared function bodies match the original syntax-tree fingerprints.
- The regenerated artifacts compile in the supplied manuscript's LaTeX environment with no overfull boxes. Figure previews and representative main/appendix table pages were visually inspected.
- The original thesis working tree and its existing modifications were preserved. No full training experiments were rerun, and no `pulseopt` implementation was changed.

Analysis packages: NumPy 2.4.3, SciPy 1.17.1, Matplotlib 3.10.9, Python 3.11.15. Training checks used pulseopt 0.1.5 and Torch 2.10.0. This documents validation of the refactor; it does not establish unknown historical training environments.

Run `make paper-check` to repeat the data and analysis checks. Install the optional `training` extra to include the training tests. Archive/file checksums and generation metadata are machine-readable in `paper/` and `reproduced_artifacts/iconip2026/`.

## Verification with the corrected paper dependency

After the author confirmed pulseopt 0.3.0, the dependency pin was corrected and the published PyPI wheel was inspected (SHA-256 `f8fa6c61a7b1bdf6e9cf4799cb2c3aa849adac47a577cac0afd9debdba552ec9`). All **26 tests passed** with that package, including the existing numerical and NLP-update regressions and a new CIFAR component-construction check covering every paper policy and both optimizers. The deprecated optimizer wrappers remain available in 0.3.0; their deprecation warnings do not indicate test failures. The results ZIP and its checksum are unchanged.
