/*
  This file is part of MADNESS.

  Test: AtomicBasisSet loading for all installed basis files.

  Verifies:
  - Each installed basis loads without error (nbf > 0 for H2O).
  - BF counts for H and O match expected values for selected bases.
  - Loading a nonexistent basis throws an exception.
*/

#include <madness/mra/mra.h>
#include <madness/chem/molecularbasis.h>
#include <madness/chem/molecule.h>
#include <madness/world/test_utilities.h>

using namespace madness;

// H2O in atomic units
static Molecule make_water() {
    Molecule mol;
    mol.add_atom(0.0,  0.0,      0.2172, 8.0, 8);  // O
    mol.add_atom(0.0,  1.4305,  -0.8688, 1.0, 1);  // H
    mol.add_atom(0.0, -1.4305,  -0.8688, 1.0, 1);  // H
    return mol;
}

// nbf for a single atom of the given atomic number using this basis
static int nbf_for_element(const AtomicBasisSet& aobasis, int atn) {
    Molecule mol;
    mol.add_atom(0.0, 0.0, 0.0, static_cast<double>(atn), atn);
    std::vector<int> at_to_bf, at_nbf;
    aobasis.atoms_to_bfn(mol, at_to_bf, at_nbf);
    return at_nbf[0];
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    World& world = World::get_default();

    test_output tout("test_aobasis");
    int result = 0;

    const Molecule water = make_water();

    // --- 1. Load every installed basis on water and assert nbf > 0 ---
    const std::vector<std::string> installed_bases = {
        "sto-3g", "sto-6g",
        "3-21g",
        "6-31g", "6-31gs", "6-31gss", "6-31++gss",
        "6-311gss", "6-311++gss",
        "cc-pvdz", "cc-pvtz",
        "aug-cc-pvdz", "aug-cc-pvtz",
        "def2-svp", "def2-tzvp", "def2-tzvpp",
    };

    for (const auto& bname : installed_bases) {
        bool ok = false;
        try {
            AtomicBasisSet b; b.read_file(bname);
            ok = (b.nbf(water) > 0);
            if (!ok && world.rank() == 0)
                print("FAIL: basis", bname, "gave nbf=0 for water");
        } catch (const std::exception& e) {
            if (world.rank() == 0)
                print("FAIL: basis", bname, "threw:", e.what());
        } catch (...) {
            if (world.rank() == 0)
                print("FAIL: basis", bname, "threw unknown exception");
        }
        tout.checkpoint(ok, "load " + bname);
        if (!ok) result = 1;
    }

    // --- 2. Spot-check BF counts for selected bases ---
    // sto-3g: H=1 S shell → 1 BF; O=3 shells (S,S,P) → 5 BF
    {
        AtomicBasisSet b; b.read_file("sto-3g");
        bool ok = (nbf_for_element(b, 1) == 1) && (nbf_for_element(b, 8) == 5);
        tout.checkpoint(ok, "sto-3g: H=1 bf, O=5 bf");
        if (!ok) result = 1;
    }
    // 6-31g: H=2 BF; O=[3s,2p] = 9 BF
    {
        AtomicBasisSet b; b.read_file("6-31g");
        bool ok = (nbf_for_element(b, 1) == 2) && (nbf_for_element(b, 8) == 9);
        tout.checkpoint(ok, "6-31g: H=2 bf, O=9 bf");
        if (!ok) result = 1;
    }
    // cc-pvdz (Cartesian): H=[2s,1p]=5 BF; O=[3s,2p,1d]=15 BF (Cartesian d=6)
    {
        AtomicBasisSet b; b.read_file("cc-pvdz");
        bool h_ok = (nbf_for_element(b, 1) == 5);
        bool o_ok = (nbf_for_element(b, 8) == 15);
        tout.checkpoint(h_ok, "cc-pvdz: H=5 bf");
        tout.checkpoint(o_ok, "cc-pvdz: O=15 bf");
        if (!h_ok || !o_ok) result = 1;
    }
    // def2-svp (Cartesian): H=[2s,1p]=5 BF; O=[3s,2p,1d]=15 BF (Cartesian d=6)
    {
        AtomicBasisSet b; b.read_file("def2-svp");
        bool h_ok = (nbf_for_element(b, 1) == 5);
        bool o_ok = (nbf_for_element(b, 8) == 15);
        tout.checkpoint(h_ok, "def2-svp: H=5 bf");
        tout.checkpoint(o_ok, "def2-svp: O=15 bf");
        if (!h_ok || !o_ok) result = 1;
    }

    // --- 3. Negative test: loading a nonexistent basis must throw ---
    {
        bool threw = false;
        try {
            AtomicBasisSet b; b.read_file("nonexistent-basis-xyz");
        } catch (...) {
            threw = true;
        }
        tout.checkpoint(threw, "nonexistent basis throws");
        if (!threw) result = 1;
    }

    finalize();
    return result;
}
