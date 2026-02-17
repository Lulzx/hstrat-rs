use alloc::vec::Vec;
use crate::column::{HereditaryStratigraphicColumn, Stratum};
use crate::differentia::Differentia;
use crate::errors::{HstratError, Result};
use crate::policies::StratumRetentionPolicy;

/// Default byte width for encoding num_strata_deposited in packets.
const DEFAULT_NUM_STRATA_DEPOSITED_BYTE_WIDTH: usize = 4;

/// Serialize a column to a binary packet, compatible with Python hstrat's
/// `col_to_packet()`.
///
/// Packet format:
/// 1. num_strata_deposited as big-endian unsigned int (4 bytes default)
/// 2. Packed differentiae at `bit_width` bits each, MSB-first, padded to byte
pub fn col_to_packet<P: StratumRetentionPolicy>(
    column: &HereditaryStratigraphicColumn<P>,
) -> Vec<u8> {
    col_to_packet_with_options(column, DEFAULT_NUM_STRATA_DEPOSITED_BYTE_WIDTH)
}

/// Serialize with configurable header width.
pub fn col_to_packet_with_options<P: StratumRetentionPolicy>(
    column: &HereditaryStratigraphicColumn<P>,
    num_strata_deposited_byte_width: usize,
) -> Vec<u8> {
    let n = column.get_num_strata_deposited();
    let bit_width = column.get_stratum_differentia_bit_width();

    // Header: num_strata_deposited in big-endian
    let mut packet = Vec::new();
    let n_bytes = n.to_be_bytes();
    // Take the last `num_strata_deposited_byte_width` bytes
    let start = 8usize.saturating_sub(num_strata_deposited_byte_width);
    packet.extend_from_slice(&n_bytes[start..]);

    // Pack differentiae
    let differentiae: Vec<u64> = column
        .iter_retained_differentia()
        .map(|d| d.value())
        .collect();
    let packed = pack_differentiae_bytes(&differentiae, bit_width);
    packet.extend_from_slice(&packed);

    packet
}

/// Deserialize a column from a binary packet.
pub fn col_from_packet<P: StratumRetentionPolicy>(
    packet: &[u8],
    policy: P,
    differentia_bit_width: u8,
) -> Result<HereditaryStratigraphicColumn<P>> {
    col_from_packet_with_options(
        packet,
        policy,
        differentia_bit_width,
        DEFAULT_NUM_STRATA_DEPOSITED_BYTE_WIDTH,
    )
}

/// Deserialize with configurable header width.
pub fn col_from_packet_with_options<P: StratumRetentionPolicy>(
    packet: &[u8],
    policy: P,
    differentia_bit_width: u8,
    num_strata_deposited_byte_width: usize,
) -> Result<HereditaryStratigraphicColumn<P>> {
    if packet.len() < num_strata_deposited_byte_width {
        return Err(HstratError::DeserializationError(
            "packet too short for header".into(),
        ));
    }

    // Read num_strata_deposited from big-endian header
    let mut n_bytes = [0u8; 8];
    let start = 8 - num_strata_deposited_byte_width;
    n_bytes[start..].copy_from_slice(&packet[..num_strata_deposited_byte_width]);
    let num_strata_deposited = u64::from_be_bytes(n_bytes);

    // Calculate expected number of retained strata
    let num_retained =
        policy.calc_num_strata_retained_exact(num_strata_deposited) as usize;

    // Unpack differentiae from remaining bytes
    let diff_bytes = &packet[num_strata_deposited_byte_width..];
    let differentiae =
        unpack_differentiae_bytes(diff_bytes, differentia_bit_width, num_retained)?;

    // Reconstruct retained ranks using the policy
    let retained_ranks: Vec<u64> = policy
        .iter_retained_ranks(num_strata_deposited)
        .collect();

    if retained_ranks.len() != differentiae.len() {
        return Err(HstratError::DeserializationError(alloc::format!(
            "rank count {} != differentia count {}",
            retained_ranks.len(),
            differentiae.len()
        )));
    }

    // Build strata
    let strata: Vec<Stratum> = retained_ranks
        .into_iter()
        .zip(differentiae.into_iter())
        .map(|(rank, diff_val)| Stratum {
            rank,
            differentia: Differentia::new(diff_val, differentia_bit_width),
        })
        .collect();

    Ok(HereditaryStratigraphicColumn::from_parts(
        policy,
        differentia_bit_width,
        strata,
        num_strata_deposited,
    ))
}

/// Pack differentia values into a byte array at the given bit width.
fn pack_differentiae_bytes(differentiae: &[u64], bit_width: u8) -> Vec<u8> {
    if differentiae.is_empty() {
        return Vec::new();
    }

    let total_bits = differentiae.len() * bit_width as usize;
    let total_bytes = (total_bits + 7) / 8;
    let mut result = alloc::vec![0u8; total_bytes];

    let mut bit_offset: usize = 0;
    for &diff in differentiae {
        for bit_idx in (0..bit_width as usize).rev() {
            let bit = ((diff >> bit_idx) & 1) as u8;
            let byte_idx = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8); // MSB first
            result[byte_idx] |= bit << bit_pos;
            bit_offset += 1;
        }
    }

    result
}

/// Unpack differentia values from a byte array at the given bit width.
fn unpack_differentiae_bytes(
    bytes: &[u8],
    bit_width: u8,
    count: usize,
) -> Result<Vec<u64>> {
    let total_bits_needed = count * bit_width as usize;
    let total_bytes_needed = (total_bits_needed + 7) / 8;

    if bytes.len() < total_bytes_needed {
        return Err(HstratError::DeserializationError(alloc::format!(
            "not enough bytes: have {}, need {}",
            bytes.len(),
            total_bytes_needed
        )));
    }

    let mut result = Vec::with_capacity(count);
    let mut bit_offset: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for _ in 0..bit_width as usize {
            let byte_idx = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8);
            let bit = ((bytes[byte_idx] >> bit_pos) & 1) as u64;
            value = (value << 1) | bit;
            bit_offset += 1;
        }
        result.push(value);
    }

    Ok(result)
}

/// Serialize a column to a JSON-compatible record.
#[cfg(feature = "serde")]
pub fn col_to_records<P: StratumRetentionPolicy>(
    column: &HereditaryStratigraphicColumn<P>,
) -> serde_json::Value {
    let ranks: Vec<u64> = column.iter_retained_ranks().collect();
    let differentiae: Vec<u64> = column
        .iter_retained_differentia()
        .map(|d| d.value())
        .collect();

    serde_json::json!({
        "policy": column.get_policy().algo_identifier(),
        "num_strata_deposited": column.get_num_strata_deposited(),
        "differentia_bit_width": column.get_stratum_differentia_bit_width(),
        "differentiae": differentiae,
        "ranks": ranks,
    })
}

/// Deserialize a column from a JSON record (as produced by `col_to_records`).
///
/// The caller must provide the policy to use for the reconstructed column.
/// The record must contain `num_strata_deposited`, `differentia_bit_width`,
/// `differentiae`, and `ranks` fields.
#[cfg(feature = "serde")]
pub fn col_from_records<P: StratumRetentionPolicy>(
    record: &serde_json::Value,
    policy: P,
) -> Result<HereditaryStratigraphicColumn<P>> {
    let num_strata_deposited = record["num_strata_deposited"]
        .as_u64()
        .ok_or_else(|| HstratError::DeserializationError(
            "missing or invalid num_strata_deposited".into(),
        ))?;

    let differentia_bit_width = record["differentia_bit_width"]
        .as_u64()
        .ok_or_else(|| HstratError::DeserializationError(
            "missing or invalid differentia_bit_width".into(),
        ))? as u8;

    let differentiae: Vec<u64> = record["differentiae"]
        .as_array()
        .ok_or_else(|| HstratError::DeserializationError(
            "missing or invalid differentiae array".into(),
        ))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();

    let ranks: Vec<u64> = record["ranks"]
        .as_array()
        .ok_or_else(|| HstratError::DeserializationError(
            "missing or invalid ranks array".into(),
        ))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();

    if ranks.len() != differentiae.len() {
        return Err(HstratError::DeserializationError(alloc::format!(
            "ranks count {} != differentiae count {}",
            ranks.len(),
            differentiae.len()
        )));
    }

    let strata: Vec<Stratum> = ranks
        .into_iter()
        .zip(differentiae.into_iter())
        .map(|(rank, diff_val)| Stratum {
            rank,
            differentia: Differentia::new(diff_val, differentia_bit_width),
        })
        .collect();

    Ok(HereditaryStratigraphicColumn::from_parts(
        policy,
        differentia_bit_width,
        strata,
        num_strata_deposited,
    ))
}

/// Serialize a population of columns to records.
#[cfg(feature = "serde")]
pub fn pop_to_records<P: StratumRetentionPolicy>(
    population: &[HereditaryStratigraphicColumn<P>],
) -> Vec<serde_json::Value> {
    population.iter().map(|col| col_to_records(col)).collect()
}

/// Deserialize a population of columns from records.
#[cfg(feature = "serde")]
pub fn pop_from_records<P: StratumRetentionPolicy + Clone>(
    records: &[serde_json::Value],
    policy: P,
) -> Result<Vec<HereditaryStratigraphicColumn<P>>> {
    records.iter().map(|r| col_from_records(r, policy.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{
        PerfectResolutionPolicy, FixedResolutionPolicy, NominalResolutionPolicy,
    };

    #[test]
    fn pack_unpack_round_trip_8bit() {
        let diffs = alloc::vec![0xFF, 0x00, 0xAB, 0x42];
        let packed = pack_differentiae_bytes(&diffs, 8);
        let unpacked = unpack_differentiae_bytes(&packed, 8, 4).unwrap();
        assert_eq!(diffs, unpacked);
    }

    #[test]
    fn pack_unpack_round_trip_1bit() {
        let diffs = alloc::vec![1, 0, 1, 1, 0, 0, 1, 0];
        let packed = pack_differentiae_bytes(&diffs, 1);
        assert_eq!(packed.len(), 1); // 8 bits = 1 byte
        let unpacked = unpack_differentiae_bytes(&packed, 1, 8).unwrap();
        assert_eq!(diffs, unpacked);
    }

    #[test]
    fn pack_unpack_round_trip_4bit() {
        let diffs = alloc::vec![0xA, 0x5, 0xF, 0x0];
        let packed = pack_differentiae_bytes(&diffs, 4);
        assert_eq!(packed.len(), 2); // 16 bits = 2 bytes
        let unpacked = unpack_differentiae_bytes(&packed, 4, 4).unwrap();
        assert_eq!(diffs, unpacked);
    }

    #[test]
    fn pack_unpack_round_trip_64bit() {
        let diffs = alloc::vec![u64::MAX, 0, 42, 1234567890123456789];
        let packed = pack_differentiae_bytes(&diffs, 64);
        assert_eq!(packed.len(), 32); // 4 * 8 bytes
        let unpacked = unpack_differentiae_bytes(&packed, 64, 4).unwrap();
        assert_eq!(diffs, unpacked);
    }

    #[test]
    fn pack_unpack_round_trip_3bit() {
        // Odd bit width
        let diffs = alloc::vec![0b111, 0b010, 0b100, 0b001, 0b011];
        let packed = pack_differentiae_bytes(&diffs, 3);
        assert_eq!(packed.len(), 2); // 15 bits → 2 bytes
        let unpacked = unpack_differentiae_bytes(&packed, 3, 5).unwrap();
        assert_eq!(diffs, unpacked);
    }

    #[test]
    fn pack_unpack_empty() {
        let packed = pack_differentiae_bytes(&[], 8);
        assert!(packed.is_empty());
        let unpacked = unpack_differentiae_bytes(&[], 8, 0).unwrap();
        assert!(unpacked.is_empty());
    }

    #[test]
    fn packet_round_trip() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy,
            64,
            42,
        );
        col.deposit_strata(10);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(
            &packet,
            PerfectResolutionPolicy,
            64,
        )
        .unwrap();

        assert_eq!(
            col.get_num_strata_deposited(),
            restored.get_num_strata_deposited()
        );
        assert_eq!(
            col.get_num_strata_retained(),
            restored.get_num_strata_retained()
        );

        // Compare strata
        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            assert_eq!(a.rank, b.rank);
            assert_eq!(a.differentia, b.differentia);
        }
    }

    #[test]
    fn packet_round_trip_fixed_resolution() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(10),
            8,
            42,
        );
        col.deposit_strata(100);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(
            &packet,
            FixedResolutionPolicy::new(10),
            8,
        )
        .unwrap();

        assert_eq!(
            col.get_num_strata_deposited(),
            restored.get_num_strata_deposited()
        );
        assert_eq!(
            col.get_num_strata_retained(),
            restored.get_num_strata_retained()
        );

        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            assert_eq!(a.rank, b.rank);
            assert_eq!(a.differentia, b.differentia);
        }
    }

    #[test]
    fn packet_round_trip_nominal_resolution() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            NominalResolutionPolicy,
            64,
            123,
        );
        col.deposit_strata(50);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(
            &packet,
            NominalResolutionPolicy,
            64,
        )
        .unwrap();

        assert_eq!(
            col.get_num_strata_deposited(),
            restored.get_num_strata_deposited()
        );
        assert_eq!(
            col.get_num_strata_retained(),
            restored.get_num_strata_retained()
        );
    }

    #[test]
    fn packet_round_trip_1bit_differentia() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy,
            1,
            42,
        );
        col.deposit_strata(20);

        let packet = col_to_packet(&col);
        let restored = col_from_packet(
            &packet,
            PerfectResolutionPolicy,
            1,
        )
        .unwrap();

        assert_eq!(
            col.get_num_strata_deposited(),
            restored.get_num_strata_deposited()
        );
        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            assert_eq!(a.rank, b.rank);
            assert_eq!(a.differentia, b.differentia);
        }
    }

    #[test]
    fn packet_empty_column() {
        let col = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy,
            64,
            42,
        );
        let packet = col_to_packet(&col);
        let restored = col_from_packet(
            &packet,
            PerfectResolutionPolicy,
            64,
        )
        .unwrap();
        assert_eq!(restored.get_num_strata_deposited(), 0);
        assert_eq!(restored.get_num_strata_retained(), 0);
    }

    #[test]
    fn packet_too_short_errors() {
        let result = col_from_packet(&[0u8; 3], PerfectResolutionPolicy, 64);
        assert!(result.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn records_round_trip() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            PerfectResolutionPolicy,
            64,
            42,
        );
        col.deposit_strata(10);

        let record = col_to_records(&col);
        let restored = col_from_records(&record, PerfectResolutionPolicy).unwrap();

        assert_eq!(
            col.get_num_strata_deposited(),
            restored.get_num_strata_deposited()
        );
        assert_eq!(
            col.get_num_strata_retained(),
            restored.get_num_strata_retained()
        );

        for (a, b) in col.iter_retained_strata().zip(restored.iter_retained_strata()) {
            assert_eq!(a.rank, b.rank);
            assert_eq!(a.differentia, b.differentia);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn records_contains_expected_fields() {
        let mut col = HereditaryStratigraphicColumn::with_seed(
            FixedResolutionPolicy::new(5),
            8,
            7,
        );
        col.deposit_strata(20);

        let record = col_to_records(&col);
        assert!(record["policy"].is_string());
        assert_eq!(record["num_strata_deposited"].as_u64().unwrap(), 20);
        assert_eq!(record["differentia_bit_width"].as_u64().unwrap(), 8);
        assert!(record["differentiae"].is_array());
        assert!(record["ranks"].is_array());
        assert_eq!(
            record["differentiae"].as_array().unwrap().len(),
            record["ranks"].as_array().unwrap().len()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn pop_records_round_trip() {
        let mut pop = Vec::new();
        for seed in 0..5u64 {
            let mut col = HereditaryStratigraphicColumn::with_seed(
                PerfectResolutionPolicy,
                64,
                seed,
            );
            col.deposit_strata(10);
            pop.push(col);
        }

        let records = pop_to_records(&pop);
        assert_eq!(records.len(), 5);

        let restored = pop_from_records(&records, PerfectResolutionPolicy).unwrap();
        assert_eq!(restored.len(), 5);

        for (orig, rest) in pop.iter().zip(restored.iter()) {
            assert_eq!(
                orig.get_num_strata_deposited(),
                rest.get_num_strata_deposited()
            );
            for (a, b) in orig.iter_retained_strata().zip(rest.iter_retained_strata()) {
                assert_eq!(a.rank, b.rank);
                assert_eq!(a.differentia, b.differentia);
            }
        }
    }
}
