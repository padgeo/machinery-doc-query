use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use qdrant_client::qdrant::{ScoredPoint, Value};

pub const SOURCE_KEY: &str = "source_file";
pub const PAGE_KEY: &str = "page_number";
pub const CHUNK_KEY: &str = "chunk_index";
pub const TEXT_KEY: &str = "text";

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum RetrievedChunkError {
    #[error("the stored point has no usable {0:?} in its payload")]
    MissingField(&'static str),
    #[error("payload field {field:?} holds {value}, which is not a valid number")]
    InvalidNumber { field: &'static str, value: i64 },
}

pub struct Chunk {
    pub source_file: PathBuf, // relative
    pub page_number: usize,   // 1-indexed
    pub chunk_index: usize,   // per-file NOT per-page
    pub text: String,
    // todo: ingested_at: timestamp
}

pub struct ChunkOptions {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        ChunkOptions {
            chunk_size: 1500,
            overlap: 200,
        }
    }
}

pub struct RetrievedChunk {
    pub chunk: Chunk,
    pub score: f32,
}

impl RetrievedChunk {
    fn get_usize_value(
        payload: &HashMap<String, Value>,
        field: &'static str,
    ) -> Result<usize, RetrievedChunkError> {
        let Some(value) = payload.get(field).and_then(Value::as_integer) else {
            Err(RetrievedChunkError::MissingField(field))?
        };

        usize::try_from(value).map_err(|_| RetrievedChunkError::InvalidNumber { field, value })
    }
}

impl TryFrom<ScoredPoint> for RetrievedChunk {
    type Error = RetrievedChunkError;

    fn try_from(scored_point: ScoredPoint) -> Result<Self, Self::Error> {
        let payload = scored_point.payload;

        let source_file = payload
            .get(SOURCE_KEY)
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or(RetrievedChunkError::MissingField(SOURCE_KEY))?;

        let page_number = Self::get_usize_value(&payload, PAGE_KEY)?;
        let chunk_index = Self::get_usize_value(&payload, CHUNK_KEY)?;

        let text = payload
            .get(TEXT_KEY)
            .and_then(Value::as_str)
            .map(|s| s.to_owned())
            .ok_or(RetrievedChunkError::MissingField(TEXT_KEY))?;

        Ok(RetrievedChunk {
            chunk: Chunk {
                source_file,
                page_number,
                chunk_index,
                text,
            },
            score: scored_point.score,
        })
    }
}

pub fn chunk_pages(source_file: &Path, pages: &[&str], options: &ChunkOptions) -> Vec<Chunk> {
    let mut result: Vec<Chunk> = Vec::new();
    let mut end = 0;
    let mut cursor = 0;
    let mut chunk_idx = 0;
    let mut cut_point = 0;

    pages.iter().enumerate().for_each(|(i, p)| {
        let page_n = i + 1;
        // A blank page (empty or only whitespace) yields no chunk, but it still
        // counts as a page so later pages keep their real page numbers.
        if p.trim().is_empty() {
            return;
        }
        cursor = 0;
        cut_point = 0;
        end = 0;

        while cursor < p.len() {
            end = safe_boundary(p, cursor + options.chunk_size);

            cut_point = cursor + find_break(&p[cursor..end]);
            if cut_point.saturating_sub(options.overlap) <= cursor {
                cut_point = end;
            }

            let new_chunk = Chunk {
                source_file: PathBuf::from(source_file),
                chunk_index: chunk_idx,
                page_number: page_n,
                text: String::from(&p[cursor..cut_point]),
            };
            result.push(new_chunk);
            chunk_idx += 1;

            if end == p.len() {
                break;
            }
            cursor = cut_point.saturating_sub(options.overlap);
        }
    });
    result
}

fn safe_boundary(s: &str, byte_idx: usize) -> usize {
    let mut i = byte_idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn find_break(s: &str) -> usize {
    for seg in ["\n\n", ".\n", "."] {
        if let Some(idx) = s.rfind(seg) {
            return idx;
        }
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // ---- safe_boundary ----

    #[test]
    fn safe_boundary_within_ascii_is_unchanged() {
        let s = "Turn off the main breaker.";
        assert_eq!(safe_boundary(s, 10), 10);
    }

    #[test]
    fn safe_boundary_clamps_to_string_length() {
        let s = "short";
        assert_eq!(safe_boundary(s, 999), s.len());
    }

    #[test]
    fn safe_boundary_at_zero_stays_zero() {
        let s = "anything";
        assert_eq!(safe_boundary(s, 0), 0);
    }

    #[test]
    fn safe_boundary_walks_back_out_of_a_multibyte_char() {
        // "30°C": byte 0='3', 1='0', 2..4 = the 2-byte '°' (U+00B0), 4='C'.
        // Byte 3 sits inside '°' and is not a valid boundary.
        let s = "30\u{b0}C";
        assert_eq!(s.len(), 5);
        assert!(!s.is_char_boundary(3));
        assert_eq!(safe_boundary(s, 3), 2);
    }

    #[test]
    fn safe_boundary_already_on_a_multibyte_boundary_is_unchanged() {
        let s = "30\u{b0}C";
        assert_eq!(safe_boundary(s, 2), 2);
        assert_eq!(safe_boundary(s, 4), 4);
    }

    // ---- find_break ----

    #[test]
    fn find_break_prefers_paragraph_break_over_sentence_end() {
        let s = "First part.\n\nSecond part ends here.";
        assert_eq!(find_break(s), s.rfind("\n\n").unwrap());
    }

    #[test]
    fn find_break_falls_back_to_period_when_no_paragraph_break() {
        let s = "Open the valve. Replace the gasket.";
        assert_eq!(find_break(s), s.rfind('.').unwrap());
    }

    #[test]
    fn find_break_returns_the_last_occurrence_not_the_first() {
        let s = "One. Two. Three.";
        let last_period = s.len() - 1;
        assert_eq!(find_break(s), last_period);
    }

    #[test]
    fn find_break_returns_full_length_when_nothing_found() {
        let s = "no punctuation in here at all";
        assert_eq!(find_break(s), s.len());
    }

    // ---- chunk_pages ----
    //
    // chunk_pages currently loops forever on any page it doesn't fully
    // finish processing in its very first loop iteration - confirmed by
    // running it in isolation with a timeout. Three compounding causes, all
    // in the `while` loop in chunk_pages:
    //   1. `cut_point` is assigned `find_break(&p[cursor..end])` directly,
    //      but that return value is relative to the slice (0 = "at cursor"),
    //      while every use of `cut_point` treats it as an absolute index
    //      into `p`. Only correct when cursor == 0.
    //      Fix: `cut_point = cursor + find_break(&p[cursor..end]);`
    //   2. Nothing stops the loop once a chunk reaches the true end of the
    //      page (`end == p.len()`) - it keeps computing a new cursor that
    //      lands before the end and loops again. Needs an explicit break
    //      once `end == p.len()`.
    //   3. Even with 1 and 2 fixed, if find_break re-finds the same break
    //      point as the previous chunk (a long run of text with no "." or
    //      "\n\n" within one window), cursor never advances. Needs a
    //      forward-progress guarantee: if the break doesn't move cursor
    //      forward, hard-cut at the window edge instead.
    //
    // The #[ignore]d tests below assert the *intended* behavior and were
    // verified against a patched copy of this function with all three fixes
    // applied - they should pass as soon as the fixes land here. Remove the
    // #[ignore] attributes once they do.

    #[test]
    fn chunk_pages_empty_pages_list_returns_empty() {
        let options = ChunkOptions {
            chunk_size: 1500,
            overlap: 200,
        };
        let chunks = chunk_pages(&PathBuf::from("empty.pdf"), &[], &options);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_pages_single_blank_page_returns_empty() {
        let options = ChunkOptions {
            chunk_size: 1500,
            overlap: 200,
        };
        let page1 = "";
        let pages = vec![page1];
        let chunks = chunk_pages(&PathBuf::from("blank.pdf"), &pages, &options);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_pages_skips_whitespace_only_pages_without_shifting_page_numbers() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let page1 = "Page one text";
        let page2 = "  \n\n \t ";
        let page3 = "Page three text";
        let pages = vec![page1, page2, page3];

        let chunks = chunk_pages(&PathBuf::from("gaps.pdf"), &pages, &options);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_number, 1);
        assert_eq!(chunks[1].page_number, 3);
        assert_eq!(chunks[1].chunk_index, 1);
    }

    #[test]
    //#[ignore = "hangs: chunk_pages never stops once a chunk reaches the end of the page - see note above"]
    fn chunk_pages_text_shorter_than_chunk_size_is_one_chunk() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let page1 = "Turn off the main breaker before servicing.";
        let pages = vec![page1];
        let chunks = chunk_pages(&PathBuf::from("short.pdf"), &pages, &options);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Turn off the main breaker before servicing");
        assert_eq!(chunks[0].page_number, 1);
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    //#[ignore = "hangs: cut_point is used as an absolute index but find_break returns a relative one - see note above"]
    fn chunk_pages_keeps_warning_sentence_whole_in_one_chunk() {
        let options = ChunkOptions {
            chunk_size: 55,
            overlap: 15,
        };
        let page1 = "Step 4: Open the drain valve. WARNING: Scalding water may discharge. Step 5: Replace the gasket.";
        let pages = vec![page1];
        let chunks = chunk_pages(&PathBuf::from("boiler.pdf"), &pages, &options);
        assert!(
            chunks
                .iter()
                .any(|c| c.text.contains("WARNING: Scalding water may discharge")),
            "expected at least one chunk to contain the full warning sentence, got: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
    }

    #[test]
    //#[ignore = "hangs: see the forward-progress note above"]
    fn chunk_pages_hard_cuts_forward_when_no_break_found_in_window() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let page1: &str = &"a".repeat(200);
        let pages = vec![page1];
        let chunks = chunk_pages(&PathBuf::from("nopunct.pdf"), &pages, &options);
        assert!(chunks.len() > 1);
        let total_chars: usize = chunks.iter().map(|c| c.text.len()).sum();
        assert!(
            total_chars >= 200,
            "chunks should cover the full 200-char page, got {total_chars} total chars"
        );
    }

    #[test]
    //#[ignore = "hangs: see the note above"]
    fn chunk_pages_chunk_index_is_monotonic_while_page_number_resets() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let page1 = "Page one text ends here without punctuation";
        let page2 = "Page two text also ends here";
        let pages = vec![page1, page2];
        let chunks = chunk_pages(&PathBuf::from("multi.pdf"), &pages, &options);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_number, 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].page_number, 2);
        assert_eq!(chunks[1].chunk_index, 1);
    }

    #[test]
    //#[ignore = "hangs: see the note above"]
    fn chunk_pages_does_not_panic_on_multibyte_text() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let page1 = "Calibrate the sensor to read 30\u{b0}C before running the test. Record all values in the logbook for review by the shift lead.";
        let pages = vec![page1];
        let chunks = chunk_pages(&PathBuf::from("multibyte.pdf"), &pages, &options);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains('\u{b0}')));
    }

    // ---- RetrievedChunk::try_from ----

    fn point(fields: &[(&str, Value)]) -> ScoredPoint {
        ScoredPoint {
            payload: fields
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect(),
            score: 0.75,
            ..Default::default()
        }
    }

    fn full_payload() -> Vec<(&'static str, Value)> {
        vec![
            (SOURCE_KEY, Value::from("boiler.pdf")),
            (PAGE_KEY, Value::from(3_i64)),
            (CHUNK_KEY, Value::from(7_i64)),
            (TEXT_KEY, Value::from("Open the drain valve.")),
        ]
    }

    // The full payload with one field's value replaced.
    fn payload_with(field: &str, value: Value) -> Vec<(&'static str, Value)> {
        full_payload()
            .into_iter()
            .map(|(key, original)| {
                if key == field {
                    (key, value.clone())
                } else {
                    (key, original)
                }
            })
            .collect()
    }

    #[test]
    fn a_complete_point_converts_with_every_field() {
        let retrieved = RetrievedChunk::try_from(point(&full_payload())).unwrap();

        assert_eq!(retrieved.chunk.source_file, PathBuf::from("boiler.pdf"));
        assert_eq!(retrieved.chunk.page_number, 3);
        assert_eq!(retrieved.chunk.chunk_index, 7);
        assert_eq!(retrieved.chunk.text, "Open the drain valve.");
        assert_eq!(retrieved.score, 0.75);
    }

    // A missing field is an error that names it. A silent default (no file name, page
    // 0, empty text) would reach the user later as if it were real data.
    #[rstest]
    #[case::source_file(SOURCE_KEY)]
    #[case::page_number(PAGE_KEY)]
    #[case::chunk_index(CHUNK_KEY)]
    #[case::text(TEXT_KEY)]
    fn a_missing_field_is_named_in_the_error(#[case] missing: &'static str) {
        let fields: Vec<_> = full_payload()
            .into_iter()
            .filter(|(key, _)| *key != missing)
            .collect();

        let err = RetrievedChunk::try_from(point(&fields))
            .err()
            .expect("a point with a missing field should be refused");

        assert_eq!(err, RetrievedChunkError::MissingField(missing));
    }

    // usize::try_from(-1) used to be unwrapped, which panics.
    #[rstest]
    #[case::page_number(PAGE_KEY)]
    #[case::chunk_index(CHUNK_KEY)]
    fn a_negative_number_is_an_error_not_a_panic(#[case] field: &'static str) {
        let fields = payload_with(field, Value::from(-1_i64));

        let err = RetrievedChunk::try_from(point(&fields))
            .err()
            .expect("a negative number should be refused");

        assert_eq!(err, RetrievedChunkError::InvalidNumber { field, value: -1 });
    }

    // A value of another type is as good as no value.
    #[rstest]
    #[case::source_file_as_number(SOURCE_KEY, Value::from(1_i64))]
    #[case::page_number_as_text(PAGE_KEY, Value::from("three"))]
    #[case::chunk_index_as_text(CHUNK_KEY, Value::from("seven"))]
    #[case::text_as_number(TEXT_KEY, Value::from(5_i64))]
    fn a_field_of_the_wrong_type_counts_as_missing(
        #[case] field: &'static str,
        #[case] wrong: Value,
    ) {
        let fields = payload_with(field, wrong);

        let err = RetrievedChunk::try_from(point(&fields))
            .err()
            .expect("a value of the wrong type should be refused");

        assert_eq!(err, RetrievedChunkError::MissingField(field));
    }
}
