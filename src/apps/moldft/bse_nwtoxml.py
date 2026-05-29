#!/usr/bin/env python3
"""Convert a BSE NWChem-format orbital basis file to MADNESS XML.

Usage:
    python3 bse_nwtoxml.py <input.nw> [basis_name]

The basis name defaults to the value found in the '# Basis set:' comment line
of the NWChem file.  The MADNESS XML is written to stdout.

General contractions (multiple coefficient columns in one element/shell block)
are reduced to segmented form: each primitive is assigned to the rightmost
contraction column with a non-zero coefficient.  This is the same approximation
used by the existing MADNESS basis files (e.g. aug-cc-pvdz).

SP shells are emitted as type="L" with <scoefficients> and <pcoefficients>,
matching the MADNESS loader in molecularbasis.cc.
"""

import sys
import re


def parse_nwchem(lines):
    """Return list of (symbol, shell_type, [(exp, [coeffs])]) from BSE NWChem."""
    basis_name = None
    shells = []          # list of (symbol, shell_type, primitives)
    in_basis = False
    current_elem = None
    current_type = None
    current_prims = []   # list of (exponent, [coeff, ...])

    for raw in lines:
        line = raw.strip()

        # Extract basis name from BSE header comment
        if basis_name is None and line.startswith('#') and 'Basis set:' in line:
            m = re.search(r'Basis set:\s*(\S+.*)', line)
            if m:
                basis_name = m.group(1).strip()

        if line.startswith('#') or not line:
            continue

        upper = line.upper()

        # BASIS ... PRINT line — marks start of data
        if upper.startswith('BASIS'):
            in_basis = True
            continue

        if upper.startswith('END'):
            if current_elem and current_type and current_prims:
                shells.append((current_elem, current_type, current_prims))
            current_elem = current_type = None
            current_prims = []
            in_basis = False
            continue

        if not in_basis:
            continue

        # Element / shell-type line: first token is an element symbol (letters only),
        # second token is the shell type (S, P, D, F, G, SP, L, …).
        tokens = line.split()
        if len(tokens) >= 2 and re.match(r'^[A-Za-z]+$', tokens[0]) and \
                re.match(r'^[SPDFGLspdfgl]+$', tokens[1]):
            # Save any open shell
            if current_elem and current_type and current_prims:
                shells.append((current_elem, current_type, current_prims))
            current_elem = tokens[0].capitalize()
            current_type = tokens[1].upper()
            current_prims = []
            continue

        # Coefficient line
        try:
            nums = [float(t) for t in tokens]
            if len(nums) >= 2:
                exp = nums[0]
                coeffs = nums[1:]
                current_prims.append((exp, coeffs))
        except ValueError:
            pass

    # Flush last shell
    if current_elem and current_type and current_prims:
        shells.append((current_elem, current_type, current_prims))

    return basis_name, shells


def split_general_contraction(prims, shell_type):
    """Split general-contraction primitives into segmented contracted shells.

    Each primitive is assigned to the rightmost column with a non-zero coefficient.
    Returns list of (exponents, coefficients_list) pairs, one per contracted function.
    For SP shells the coefficients_list has two entries [s_coeffs, p_coeffs].
    """
    if not prims:
        return []

    n_cols = max(len(c) for _, c in prims)

    if shell_type == 'SP':
        # SP has exactly 2 columns (s-coeff, p-coeff); treat as single contracted function
        exps = [e for e, _ in prims]
        s_coeffs = [c[0] if len(c) > 0 else 0.0 for _, c in prims]
        p_coeffs = [c[1] if len(c) > 1 else 0.0 for _, c in prims]
        return [(exps, [s_coeffs, p_coeffs])]

    # For regular shells: for each contraction column, include all primitives
    # where that column has a non-zero coefficient (full general contraction).
    # Skip columns that are entirely zero.
    result = []
    for col in range(n_cols):
        exps = []
        coeffs_for_col = []
        for exp, coeffs in prims:
            padded = list(coeffs) + [0.0] * (n_cols - len(coeffs))
            if padded[col] != 0.0:
                exps.append(exp)
                coeffs_for_col.append(padded[col])
        if exps:
            result.append((exps, [coeffs_for_col]))
    return result


def format_float(v):
    """Format a basis-set coefficient/exponent for XML output."""
    # Use scientific notation for very small/large values, fixed otherwise
    if v == 0.0:
        return '      0.00000000'
    return '      %.8f' % v


def shells_to_xml(basis_name, shells):
    lines = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<name>')
    lines.append('   %s' % basis_name)
    lines.append('</name>')

    # Group by element symbol
    from collections import OrderedDict
    by_elem = OrderedDict()
    for sym, stype, prims in shells:
        by_elem.setdefault(sym, []).append((stype, prims))

    for sym, elem_shells in by_elem.items():
        lines.append('<basis symbol="%s">' % sym)
        for stype, prims in elem_shells:
            contracted = split_general_contraction(prims, stype)
            for exps, coeff_sets in contracted:
                nprim = len(exps)
                xml_type = 'L' if stype == 'SP' else stype
                lines.append('  <shell type="%s" nprim="%d">' % (xml_type, nprim))
                lines.append('    <exponents>')
                for e in exps:
                    lines.append(format_float(e))
                lines.append('    </exponents>')
                if xml_type == 'L':
                    lines.append('    <scoefficients>')
                    for c in coeff_sets[0]:
                        lines.append(format_float(c))
                    lines.append('    </scoefficients>')
                    lines.append('    <pcoefficients>')
                    for c in coeff_sets[1]:
                        lines.append(format_float(c))
                    lines.append('    </pcoefficients>')
                else:
                    lines.append('    <coefficients>')
                    for c in coeff_sets[0]:
                        lines.append(format_float(c))
                    lines.append('    </coefficients>')
                lines.append('  </shell>')
        lines.append('</basis>')

    return '\n'.join(lines) + '\n'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    nwfile = sys.argv[1]
    with open(nwfile) as f:
        lines = f.readlines()

    detected_name, shells = parse_nwchem(lines)

    if len(sys.argv) >= 3:
        basis_name = sys.argv[2]
    elif detected_name:
        basis_name = detected_name
    else:
        basis_name = 'unknown'

    print(shells_to_xml(basis_name, shells), end='')


if __name__ == '__main__':
    main()
