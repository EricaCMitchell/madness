# `moldft` — quick guide

A MADNESS app for molecular Hartree–Fock and DFT in a real-space
multiresolution (MRA) basis. Single executable, text input file.

## What it does

- Solves SCF (RHF/UHF or KS-DFT) for a finite molecule on a numerical
  MRA grid — no Gaussian basis set; accuracy is controlled by `k` and
  `thresh`.
- Outputs total energy, dipole, optionally gradients, orbital plots,
  and a JSON `calc_info` schema for downstream tooling.
- Supports geometry optimization (`MolOpt`) when `gopt` is set.
- XC functionals via libxc (LDA, GGA, hybrids like PBE0, plain HF).

## Source layout (`src/apps/moldft/`)

- `moldft.cc` — thin driver (`main`); all real work lives in
  `madness::SCF` under `src/madness/chem/SCF.{h,cc}`.
- `subspace.h`, `wst_functional.h`, `pointgroup.h` — local helpers used
  by the driver.
- `tests/*.in` — small reference inputs (atoms + small molecules,
  HF and LDA) with expected energies in the file header comments.
- `input` — large multi-block example (water clusters etc.).
- `testpg.cc`, `testperiodicdft.cc` — auxiliary test executables built
  alongside (`testpg`, `testperiodicdft`).
- `mcpfit.cc`, `preal.cc`, `testcosine.cc`, `testmolbas.cc` — historic;
  not built (commented out in `CMakeLists.txt`).

### `src/madness/chem/` — AO basis tests

- `test_aobasis.cc` — unit test for `AtomicBasisSet` load/eval correctness
  across the installed bases; registered as a `short` CTest entry.

## Building

From a configured build directory (see top-level `CLAUDE.md` /
`INSTALL.md` for configure):

```
ninja moldft
```

Binary lands at `<build-dir>/src/apps/moldft/moldft`. Install target
is also wired (`install(TARGETS moldft …)`).

## Running

Serial:

```
./moldft --input=tests/he_hf.in
```

MPI (recommended for anything beyond a few atoms; see `CLAUDE.md`
for thread/pinning rules):

```
export MAD_NUM_THREADS=2
mpiexec -np 4 ./moldft --input=h2o.in
```

Default input filename is `input` if `--input=` is omitted.

### CLI options

- `--input=<file>` — input deck (default `input`).
- `--help` — print the full SCF parameter list with defaults.
- `--print_parameters` — dump current parameter values then exit.
- `--<block>="key=val; key=val"` — override any input-deck block on
  the command line. Used by the smoke command in `CLAUDE.md`:

  ```
  ./moldft --geometry=water --dft="xc=lda"
  ```

  Here `--geometry=water` selects a built-in geometry preset and
  `--dft="…"` overrides the `dft` block.

## Input file format

Free-form text, blocks delimited by `<name> … end`. Lines starting with
`#` are comments. Two blocks matter in practice:

### `dft` block — SCF parameters

```
dft
  xc lda                # or: hf, b3lyp, "GGA_X_PBE 1.0 GGA_C_PBE 1.0", …
  maxsub 5              # KAIN subspace size
  # econv, dconv, k, protocol, nvalpha, charge, gopt, derivatives, …
end
```

Run `./moldft --help` or `--print_parameters` for the complete list
(it's long — pulled from `CalculationParameters` in
`src/madness/chem/SCF.h`).

### `geometry` block

```
geometry
  units angstrom        # default is atomic units (bohr)
  eprec 1e-7            # nuclear-potential smoothing
  O  0.0     0.0  0.0
  H  1.4375  0.0  1.15
  H -1.4375  0.0  1.15
end
```

Multiple `geometry` blocks may sit in one file; the parser uses the
first one (others are effectively ignored — comment them out for
clarity). See `tests/h2o_1_lda.in` for a minimal working example.

## Output

- Header echoes molecule, parameters, and MADNESS revision.
- SCF iterations log energy and convergence per protocol step
  (`protocol` is a sequence of tightening thresholds).
- On success: `final energy=…`, dipole, optional gradient.
- A `calc_info.json` file (schema in
  `mad_moldft_test_energy*.ref.json`) summarizes the run for
  scripted regression checks.
- Orbital / density cube plots are written if requested via the
  `plot` parameters (`calc.do_plots`).

## Regression / smoke checks

- Quick correctness check (~seconds):

  ```
  ./moldft --input=src/apps/moldft/tests/he_hf.in
  ```

  Reference energies are in the header comment of each `.in`.
  `tests/h2o_ccpvdz_lda.in` exercises `aobasis cc-pvdz` specifically.

- Wider numerical regression: the MRA `testsuite` binary
  (`src/madness/mra/testsuite`) — see top-level `CLAUDE.md`.

- The Python harnesses `mad_moldft_test_energy*.py` and their
  `*.ref.json` companions are present but not currently registered
  with CTest (see commented `add_scripted_tests` in
  `CMakeLists.txt`); invoke them manually if needed.

## Supported AO bases (`aobasis`)

The `aobasis` parameter selects the atom-centered Gaussian basis used only
for constructing the **initial-guess density**.  The MRA grid (`k`, `thresh`)
controls all subsequent SCF accuracy — `aobasis` does not affect the
converged result.

### Installed bases

| Name | Family | Quality |
|------|--------|---------|
| `sto-3g` | Minimal | SZ |
| `sto-6g` | Minimal | SZ |
| `3-21g` | Pople | DZ |
| `6-31g` | Pople | DZ (default) |
| `6-31gs` | Pople | DZP |
| `6-31gss` | Pople | DZP |
| `6-31++gss` | Pople | DZP + diffuse |
| `6-311gss` | Pople | TZP |
| `6-311++gss` | Pople | TZP + diffuse |
| `cc-pvdz` | Dunning | DZ |
| `cc-pvtz` | Dunning | TZ |
| `aug-cc-pvdz` | Dunning | DZ + diffuse |
| `aug-cc-pvtz` | Dunning | TZ + diffuse |
| `def2-svp` | Karlsruhe | DZ |
| `def2-tzvp` | Karlsruhe | TZ |
| `def2-tzvpp` | Karlsruhe | TZP |

XML files live in `src/madness/chem/basissets/` (build tree) and
`${MADNESS_INSTALL_DATADIR}/basissets` (install tree). The runtime locates
them via the compiled-in `MRA_CHEMDATA_DIR` constant (points to the
`basissets/` subdirectory), overridden at runtime by the environment variable
`MRA_CHEMDATA_DIR`.

The `structure_library` molecule presets use a separate compiled-in constant
`MRA_CHEMDATA_ROOT` (points to `src/madness/chem/`), overridden by the
environment variable `MRA_CHEMDATA_ROOT`. The two env vars are independent —
set `MRA_CHEMDATA_DIR` to change where basis files are found, and
`MRA_CHEMDATA_ROOT` to change where `structure_library` is found.

### Custom basis via `nwfile`

To use a basis not in the list above, write it in NWChem format and set:

```
dft
  nwfile /path/to/molecule_with_basis.nw
end
```

The file must contain both the geometry and the `BASIS` block.
`src/apps/moldft/bse_nwtoxml.py` converts BSE NWChem-format files to
MADNESS XML if you want to add a new basis permanently.

### Fallback behaviour

If the selected `aobasis` file has no embedded atomic guess densities
(all new BSE-converted bases lack them), moldft automatically uses
`sto-3g` densities for the initial density estimate while still
projecting the AO functions from the requested basis.  The fallback is
always logged by rank 0.  For any element not covered by sto-3g either,
a trivial normalized spherical Gaussian integrating to Z electrons is
used as a secondary fallback.

### Error diagnostics

If a basis file cannot be loaded, moldft prints the directory searched
and lists the available basis files found there (up to 20), making it
easier to spot typos or a missing installation.

## Gotchas

- **No Gaussian basis.** "Basis" in moldft means the MRA polynomial
  order `k` and truncation threshold `thresh`; see `dft` parameters.
- **Sequential BLAS only** (project-wide MADNESS rule).
- **Geometry optimization** runs via `MolOpt` when `gopt` is true; the
  older `QuasiNewton` path in `moldft.cc` is commented out.
- **Default `World`** is used in the driver — fine here because moldft
  is top-level application code, but don't copy this pattern into
  library code (`CLAUDE.md` covers why).
- Large-`k` / 6D-style runs may need a bigger `MAD_BUFFER_SIZE` (see
  top-level `CLAUDE.md`).

## Where to dig deeper

- Driver: `src/apps/moldft/moldft.cc` (≈230 lines).
- SCF engine, parameter list, helpers:
  `src/madness/chem/SCF.{h,cc}`, `molecule.{h,cc}`,
  `molecularbasis.{h,cc}` (atomic guess only),
  `xcfunctional.{h,cc}`.
- Geometry optimizer: `src/madness/chem/MolecularOptimizer.h`.
