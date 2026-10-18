use std::path::PathBuf;

use qdrant_client::qdrant::ScoredPoint;

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

#[allow(unused)] // Will be used in the future
pub struct RetrievedChunk {
    chunk: Chunk,
    score: f32,
}

impl RetrievedChunk {
    pub fn from_scored_point(scored_point: ScoredPoint) -> Self {
        let payload = scored_point.payload;

        let source_file: PathBuf = match payload.get("source_file") {
            None => PathBuf::new(),
            Some(payload_value) => match payload_value.as_str() {
                None => PathBuf::new(),
                Some(payload_str_value) => PathBuf::from(payload_str_value),
            },
        };

        let page_number = match payload.get("page_number") {
            None => 0,
            Some(payload_value) => match payload_value.as_integer() {
                None => 0,
                Some(payload_int_value) => match usize::try_from(payload_int_value) {
                    Err(_) => 0,
                    Ok(payload_usize_value) => payload_usize_value,
                },
            },
        };

        let chunk_index = match payload.get("chunk_index") {
            None => 0,
            Some(payload_value) => match payload_value.as_integer() {
                None => 0,
                Some(payload_int_value) => match usize::try_from(payload_int_value) {
                    Err(_) => 0,
                    Ok(payload_usize_value) => payload_usize_value,
                },
            },
        };

        let text = match payload.get("text") {
            None => String::from(""),
            Some(payload_value) => match payload_value.as_str() {
                None => String::from(""),
                Some(payload_str_value) => String::from(payload_str_value),
            },
        };

        RetrievedChunk {
            chunk: Chunk {
                source_file,
                page_number,
                chunk_index,
                text,
            },
            score: scored_point.score,
        }
    }
}

pub fn chunk_pages(source_file: &PathBuf, pages: &[String], options: &ChunkOptions) -> Vec<Chunk> {
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
                source_file: source_file.clone(),
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
        let pages = vec![String::new()];
        let chunks = chunk_pages(&PathBuf::from("blank.pdf"), &pages, &options);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_pages_skips_whitespace_only_pages_without_shifting_page_numbers() {
        let options = ChunkOptions {
            chunk_size: 50,
            overlap: 15,
        };
        let pages = vec![
            "Page one text".to_string(),
            "  \n\n \t ".to_string(),
            "Page three text".to_string(),
        ];

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
        let pages = vec!["Turn off the main breaker before servicing.".to_string()];
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
        let pages = vec![
            "Step 4: Open the drain valve. WARNING: Scalding water may discharge. Step 5: Replace the gasket."
                .to_string(),
        ];
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
        let pages = vec!["a".repeat(200)];
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
        let pages = vec![
            "Page one text ends here without punctuation".to_string(),
            "Page two text also ends here".to_string(),
        ];
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
        let pages = vec![
            "Calibrate the sensor to read 30\u{b0}C before running the test. Record all values in the logbook for review by the shift lead."
                .to_string(),
        ];
        let chunks = chunk_pages(&PathBuf::from("multibyte.pdf"), &pages, &options);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.text.contains('\u{b0}')));
    }
}
