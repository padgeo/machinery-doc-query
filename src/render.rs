use std::iter::once;

use crate::chunk::{Chunk, RetrievedChunk};

fn render_one(chunk: &Chunk, score: f32, rank: usize) -> String {
    let header = format!(
        "{rank}. {}, page {}, score {score:.2}",
        chunk.source_file.display(),
        chunk.page_number
    );
    // trim() drops blank lines at the start and end of the text. Blank lines in the
    // middle stay, but empty: no line is left holding only spaces.
    let body = chunk.text.trim().lines().map(|line| {
        let line = line.trim_end();
        if line.is_empty() {
            String::new()
        } else {
            format!("   {line}")
        }
    });
    once(header).chain(body).collect::<Vec<_>>().join("\n")
}

/// Renders ranked results as plain text: a header line per result (rank, file,
/// page, score) followed by its text indented by three spaces, with a blank
/// line between results. With no results it returns a sentence saying so.
pub fn render_chunks(chunks: &[RetrievedChunk]) -> String {
    if chunks.is_empty() {
        "No matching passages found.".to_owned()
    } else {
        chunks
            .iter()
            .zip(1..)
            .map(|(r_chunk, rank)| render_one(&r_chunk.chunk, r_chunk.score, rank))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use super::*;

    fn hit(file: &str, page: usize, text: &str, score: f32) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk {
                source_file: PathBuf::from(file),
                page_number: page,
                chunk_index: 0,
                text: text.to_string(),
            },
            score,
        }
    }

    #[test]
    fn no_results_give_a_sentence_not_an_empty_string() {
        assert_eq!(render_chunks(&[]), "No matching passages found.");
    }

    #[test]
    fn one_result_is_a_header_line_and_its_indented_text() {
        let results = [hit(
            "boiler_x200_manual.pdf",
            3,
            "Open the drain valve.",
            0.75,
        )];

        assert_eq!(
            render_chunks(&results),
            "1. boiler_x200_manual.pdf, page 3, score 0.75\n   Open the drain valve."
        );
    }

    // Ranks count up from 1 in the order given, and a blank line separates the results.
    #[test]
    fn results_are_numbered_from_one_and_separated_by_a_blank_line() {
        let results = [
            hit("boiler.pdf", 3, "Open the drain valve.", 0.75),
            hit("pump.pdf", 1, "Check the seal.", 0.5),
        ];
        let expected = [
            "1. boiler.pdf, page 3, score 0.75",
            "   Open the drain valve.",
            "",
            "2. pump.pdf, page 1, score 0.50",
            "   Check the seal.",
        ]
        .join("\n");

        assert_eq!(render_chunks(&results), expected);
    }

    // A warning that follows the step it belongs to has to stay visible, so every
    // line of the text is shown, each one indented.
    #[test]
    fn every_line_of_a_multi_line_text_is_indented() {
        let results = [hit(
            "boiler.pdf",
            3,
            "Open the drain valve.\nWARNING: scalding water.\nClose it again.",
            0.75,
        )];
        let expected = [
            "1. boiler.pdf, page 3, score 0.75",
            "   Open the drain valve.",
            "   WARNING: scalding water.",
            "   Close it again.",
        ]
        .join("\n");

        assert_eq!(render_chunks(&results), expected);
    }

    // Text extracted from a PDF can end its lines with \r\n or end with a newline.
    // Neither may leave a stray \r or an extra indented blank line behind.
    #[rstest]
    #[case::lf("first\nsecond")]
    #[case::crlf("first\r\nsecond")]
    #[case::trailing_newline("first\nsecond\n")]
    fn line_endings_in_the_text_make_no_difference(#[case] text: &str) {
        let results = [hit("a.pdf", 1, text, 0.5)];

        assert_eq!(
            render_chunks(&results),
            "1. a.pdf, page 1, score 0.50\n   first\n   second"
        );
    }

    // Blank lines are indented like every other line, so this one holds three spaces.
    // If you decide to keep blank lines empty, change this test and render_one.
    #[test]
    fn a_blank_line_inside_the_text_is_indented_like_the_rest() {
        let results = [hit("a.pdf", 1, "First step.\n\nSecond step.", 0.5)];

        assert_eq!(
            render_chunks(&results),
            "1. a.pdf, page 1, score 0.50\n   First step.\n\n   Second step."
        );
    }

    // The score is similarity, shown with two decimals: Display alone would print
    // 0.8249999 and 1.
    #[rstest]
    #[case::three_quarters(0.75, "0.75")]
    #[case::one_decimal(0.5, "0.50")]
    #[case::exactly_one(1.0, "1.00")]
    #[case::long_fraction(0.824_999_9, "0.82")]
    #[case::negative(-0.5, "-0.50")]
    fn the_score_has_two_decimals(#[case] score: f32, #[case] shown: &str) {
        let results = [hit("a.pdf", 1, "text", score)];

        let header = render_chunks(&results).lines().next().unwrap().to_string();

        assert_eq!(header, format!("1. a.pdf, page 1, score {shown}"));
    }

    // The path is shown as stored, folders included: it is relative to the folder
    // that was ingested, and two manuals can share a file name.
    #[test]
    fn the_file_is_shown_with_its_folders() {
        let results = [hit("boilers/x200/manual.pdf", 12, "text", 0.5)];

        let header = render_chunks(&results).lines().next().unwrap().to_string();

        assert_eq!(header, "1. boilers/x200/manual.pdf, page 12, score 0.50");
    }

    // process_query prints the text with println!, which adds its own newline.
    #[rstest]
    #[case::one(1)]
    #[case::three(3)]
    fn the_text_has_no_trailing_newline(#[case] count: usize) {
        let results: Vec<_> = (0..count).map(|_| hit("a.pdf", 1, "text", 0.5)).collect();

        assert!(!render_chunks(&results).ends_with('\n'));
    }
}
