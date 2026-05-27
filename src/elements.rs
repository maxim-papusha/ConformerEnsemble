//! Atomic-weight lookup for atomic numbers 1..=118.
//!
//! Values are standard atomic weights in unified atomic mass units (u),
//! taken from IUPAC 2021 recommendations. Elements with no stable isotope
//! use the most-stable-isotope mass. The table is indexed by atomic
//! number (`Z`); index 0 is unused (`0.0`).

use crate::error::ConformerError;
use crate::geometry::AtomType;

/// Standard atomic weights [u], indexed by atomic number `Z` (0..=118).
/// Index 0 is a placeholder and not a valid element.
const ATOMIC_WEIGHTS: [f64; 119] = [
    0.0,        //  0 (placeholder)
    1.008,      //  1 H
    4.002602,   //  2 He
    6.94,       //  3 Li
    9.0121831,  //  4 Be
    10.81,      //  5 B
    12.011,     //  6 C
    14.007,     //  7 N
    15.999,     //  8 O
    18.998403163, //  9 F
    20.1797,    // 10 Ne
    22.98976928,// 11 Na
    24.305,     // 12 Mg
    26.9815385, // 13 Al
    28.085,     // 14 Si
    30.973761998,// 15 P
    32.06,      // 16 S
    35.45,      // 17 Cl
    39.948,     // 18 Ar
    39.0983,    // 19 K
    40.078,     // 20 Ca
    44.955908,  // 21 Sc
    47.867,     // 22 Ti
    50.9415,    // 23 V
    51.9961,    // 24 Cr
    54.938044,  // 25 Mn
    55.845,     // 26 Fe
    58.933194,  // 27 Co
    58.6934,    // 28 Ni
    63.546,     // 29 Cu
    65.38,      // 30 Zn
    69.723,     // 31 Ga
    72.630,     // 32 Ge
    74.921595,  // 33 As
    78.971,     // 34 Se
    79.904,     // 35 Br
    83.798,     // 36 Kr
    85.4678,    // 37 Rb
    87.62,      // 38 Sr
    88.90584,   // 39 Y
    91.224,     // 40 Zr
    92.90637,   // 41 Nb
    95.95,      // 42 Mo
    98.0,       // 43 Tc
    101.07,     // 44 Ru
    102.90550,  // 45 Rh
    106.42,     // 46 Pd
    107.8682,   // 47 Ag
    112.414,    // 48 Cd
    114.818,    // 49 In
    118.710,    // 50 Sn
    121.760,    // 51 Sb
    127.60,     // 52 Te
    126.90447,  // 53 I
    131.293,    // 54 Xe
    132.90545196,// 55 Cs
    137.327,    // 56 Ba
    138.90547,  // 57 La
    140.116,    // 58 Ce
    140.90766,  // 59 Pr
    144.242,    // 60 Nd
    145.0,      // 61 Pm
    150.36,     // 62 Sm
    151.964,    // 63 Eu
    157.25,     // 64 Gd
    158.92535,  // 65 Tb
    162.500,    // 66 Dy
    164.93033,  // 67 Ho
    167.259,    // 68 Er
    168.93422,  // 69 Tm
    173.045,    // 70 Yb
    174.9668,   // 71 Lu
    178.49,     // 72 Hf
    180.94788,  // 73 Ta
    183.84,     // 74 W
    186.207,    // 75 Re
    190.23,     // 76 Os
    192.217,    // 77 Ir
    195.084,    // 78 Pt
    196.966569, // 79 Au
    200.592,    // 80 Hg
    204.38,     // 81 Tl
    207.2,      // 82 Pb
    208.98040,  // 83 Bi
    209.0,      // 84 Po
    210.0,      // 85 At
    222.0,      // 86 Rn
    223.0,      // 87 Fr
    226.0,      // 88 Ra
    227.0,      // 89 Ac
    232.0377,   // 90 Th
    231.03588,  // 91 Pa
    238.02891,  // 92 U
    237.0,      // 93 Np
    244.0,      // 94 Pu
    243.0,      // 95 Am
    247.0,      // 96 Cm
    247.0,      // 97 Bk
    251.0,      // 98 Cf
    252.0,      // 99 Es
    257.0,      //100 Fm
    258.0,      //101 Md
    259.0,      //102 No
    266.0,      //103 Lr
    267.0,      //104 Rf
    268.0,      //105 Db
    269.0,      //106 Sg
    270.0,      //107 Bh
    269.0,      //108 Hs
    278.0,      //109 Mt
    281.0,      //110 Ds
    282.0,      //111 Rg
    285.0,      //112 Cn
    286.0,      //113 Nh
    289.0,      //114 Fl
    289.0,      //115 Mc
    293.0,      //116 Lv
    294.0,      //117 Ts
    294.0,      //118 Og
];

/// Returns the standard atomic weight for atomic number `z` in unified
/// atomic mass units (u, also known as Dalton or amu).
///
/// # Errors
/// Returns [`ConformerError::UnknownAtomType`] for `z == 0` or `z > 118`.
#[inline]
pub fn atomic_weight(z: AtomType) -> Result<f64, ConformerError> {
    let idx = z as usize;
    if idx == 0 || idx >= ATOMIC_WEIGHTS.len() {
        return Err(ConformerError::UnknownAtomType(z));
    }
    Ok(ATOMIC_WEIGHTS[idx])
}

/// Returns the atomic weights for each atom type in `atom_types`.
pub fn atomic_weights(atom_types: &[AtomType]) -> Result<Vec<f64>, ConformerError> {
    atom_types.iter().copied().map(atomic_weight).collect()
}
