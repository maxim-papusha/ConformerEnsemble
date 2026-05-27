use std::fmt::Write;

use ndarray::{Array2, ArrayView2, ShapeBuilder};

use crate::error::ConformerError;
use crate::geometry::AtomType;

#[derive(Debug, Clone)]
pub(crate) struct XyzBlock {
    pub(crate) atom_types: Vec<AtomType>,
    pub(crate) coords: Array2<f64>,
    pub(crate) comment: String,
}

pub(crate) fn parse_xyz_blocks(
    input: &str,
) -> Result<Vec<XyzBlock>, ConformerError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0usize;
    let mut blocks = Vec::new();

    while i < lines.len() {
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }

        let natoms_line = lines[i].trim();
        let n_atoms: usize = natoms_line.parse().map_err(|_| {
            ConformerError::XyzParse(format!(
                "line {}: invalid atom-count line '{}'; expected integer",
                i + 1,
                natoms_line
            ))
        })?;
        i += 1;

        if i >= lines.len() {
            return Err(ConformerError::XyzParse(format!(
                "line {}: missing comment line",
                i + 1
            )));
        }
        let comment = lines[i].to_string();
        i += 1;

        if i + n_atoms > lines.len() {
            return Err(ConformerError::XyzParse(format!(
                "line {}: expected {} coordinate lines, found {}",
                i + 1,
                n_atoms,
                lines.len().saturating_sub(i)
            )));
        }

        let mut atom_types = Vec::with_capacity(n_atoms);
        let mut coords = Array2::<f64>::zeros((n_atoms, 3).f());

        for row in 0..n_atoms {
            let line_no = i + row + 1;
            let line = lines[i + row];
            let mut parts = line.split_whitespace();

            let atom_token = parts.next().ok_or_else(|| {
                ConformerError::XyzParse(format!(
                    "line {}: missing atom label",
                    line_no
                ))
            })?;

            let atom_type = parse_atom_token(atom_token).ok_or_else(|| {
                ConformerError::XyzParse(format!(
                    "line {}: unknown atom label '{}'",
                    line_no, atom_token
                ))
            })?;

            let x = parse_xyz_float(parts.next(), line_no, "x")?;
            let y = parse_xyz_float(parts.next(), line_no, "y")?;
            let z = parse_xyz_float(parts.next(), line_no, "z")?;

            atom_types.push(atom_type);
            coords[[row, 0]] = x;
            coords[[row, 1]] = y;
            coords[[row, 2]] = z;
        }

        i += n_atoms;
        blocks.push(XyzBlock {
            atom_types,
            coords,
            comment,
        });
    }

    if blocks.is_empty() {
        return Err(ConformerError::XyzParse(
            "no xyz geometry blocks found".to_string(),
        ));
    }

    Ok(blocks)
}

pub(crate) fn write_xyz_block(
    output: &mut String,
    atom_types: &[AtomType],
    coords: ArrayView2<'_, f64>,
    comment: &str,
) {
    let _ = writeln!(output, "{}", atom_types.len());
    let _ = writeln!(output, "{}", comment);

    for (atom_type, row) in atom_types.iter().zip(coords.rows()) {
        let _ = writeln!(
            output,
            "{} {:.8} {:.8} {:.8}",
            atom_symbol(*atom_type),
            row[0],
            row[1],
            row[2]
        );
    }
}

/// Returns the IUPAC element symbol for the given atomic number, or a
/// decimal-formatted fallback (e.g. `"<142>"`) when the number is out of
/// range. The parser accepts both symbols and decimal labels so the
/// round-trip remains lossless.
pub(crate) fn atom_symbol(atom_type: AtomType) -> &'static str {
    let idx = atom_type as usize;
    if idx == 0 || idx >= ELEMENT_SYMBOLS.len() {
        // Unknown labels are rare (atomic numbers only span 1..=118 in
        // ELEMENT_SYMBOLS); when they happen, fall back to a debug marker
        // rather than panicking so the writer stays infallible.
        "?"
    } else {
        ELEMENT_SYMBOLS[idx]
    }
}

pub(crate) fn parse_energy_from_comment(comment: &str) -> Option<f64> {
    let trimmed = comment.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(value) = trimmed.parse::<f64>() {
        return Some(value);
    }

    for token in trimmed
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
    {
        if token.is_empty() {
            continue;
        }

        if let Some(value) = token.strip_prefix("energy=") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        if let Some(value) = token.strip_prefix("Energy=") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        if let Some(value) = token.strip_prefix("E=") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
        if let Some(value) = token.strip_prefix("e=") {
            if let Ok(parsed) = value.parse::<f64>() {
                return Some(parsed);
            }
        }
    }

    None
}

fn parse_xyz_float(
    value: Option<&str>,
    line_no: usize,
    axis: &str,
) -> Result<f64, ConformerError> {
    let raw = value.ok_or_else(|| {
        ConformerError::XyzParse(format!(
            "line {}: missing {} coordinate",
            line_no, axis
        ))
    })?;

    raw.parse::<f64>().map_err(|_| {
        ConformerError::XyzParse(format!(
            "line {}: invalid {} coordinate '{}'",
            line_no, axis, raw
        ))
    })
}

fn parse_atom_token(token: &str) -> Option<AtomType> {
    // Allow atomic numbers (1..=118) as labels.
    if let Ok(n) = token.parse::<u16>() {
        if (1..=118).contains(&n) {
            return Some(n as u8);
        }
        return None;
    }

    // Allow symbols in any case (for example `cl`, `CL`, `Cl`).
    let canonical = canonicalize_symbol(token)?;
    ELEMENT_SYMBOLS
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(idx, symbol)| {
            if *symbol == canonical {
                Some(idx as u8)
            } else {
                None
            }
        })
}

fn canonicalize_symbol(token: &str) -> Option<String> {
    if token.is_empty() || !token.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }

    let mut chars = token.chars();
    let first = chars.next()?;

    let mut canonical = String::with_capacity(token.len());
    canonical.push(first.to_ascii_uppercase());
    for c in chars {
        canonical.push(c.to_ascii_lowercase());
    }

    Some(canonical)
}

const ELEMENT_SYMBOLS: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];