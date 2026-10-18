# Quick start: water from a machine-learned initial density

`parsec.in` is a PBE calculation of water whose SCF starts from a ChargE3Net
prediction (`Initial_Density: charge3net`). The installation of the models,
the other ways to use them, and every option are described in the
[tutorial](../../../src/parsec_python/MLDensity/README.md); this folder is
the example it uses.

From the PARSEC.py checkout, with the models installed in
`parsec_ml_workspace` next to it (tutorial, section 2):

```bash
cp -r examples/ml_initial_density/quickstart ~/water   # keep results out of the checkout
PP=$PWD/examples/ml_initial_density/pseudopotentials

python src/parsec_python/ml_density.py check                           # are the models ready?
python src/parsec_python/main.py ~/water/parsec.in --pp-dir $PP        # predict, then SCF
```

or predict first and start the SCF from the file:

```bash
python src/parsec_python/ml_density.py predict --input ~/water/parsec.in --pp-dir $PP
# add the two printed lines (Initial_Density, ML_Density_File) to parsec.in, then
python src/parsec_python/main.py ~/water/parsec.in --pp-dir $PP
```

For SCDP write `Initial_Density: scdp`. Models installed elsewhere are named
by the commented path lines of `parsec.in`.
