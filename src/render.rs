use std::iter::once;

use crate::{
    answer::{Answer, AnswerStep, Source},
    chunk::{Chunk, RetrievedChunk},
};

fn render_one(chunk: &Chunk, score: f32, rank: usize) -> String {
    let header = format!(
        "{rank}. {}, page {}, score {score:.2}",
        chunk.source_file.display(),
        chunk.page_number
    );

    let body = clean_lines(&chunk.text).map(indent);
    once(header).chain(body).collect::<Vec<_>>().join("\n")
}

/// The lines of a text as they are shown: no blank lines at the start or the end (`trim`),
/// no spaces at the end of a line and no `\r` (`lines` drops it). Blank lines in the middle
/// stay.
fn clean_lines(text: &str) -> impl Iterator<Item = &str> {
    text.trim().lines().map(str::trim_end)
}

/// Indents a line by three spaces. A blank line stays empty, so that no line is left
/// holding only spaces.
fn indent(line: &str) -> String {
    if line.is_empty() {
        String::new()
    } else {
        format!("   {line}")
    }
}

/// The cleaned lines of a text joined back into one text, for texts that are not indented.
fn clean_text(text: &str) -> String {
    clean_lines(text).collect::<Vec<_>>().join("\n")
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

/// Renders an answer as plain text, ready for `println!`.
///
/// A procedure is its summary, its steps and its sources, with a blank line between
/// them:
///
/// ```text
/// Drain the boiler.
///
/// 1. Open the drain valve. [1]
///    WARNING: scalding water.
/// 2. Tighten the bolts. [no source]
///
/// Sources:
/// [1] boiler.pdf, page 3
/// ```
///
/// Steps are numbered from 1 in the order given. The numbers in brackets at the end of a
/// step are the sources it cites, and `[no source]` marks a step that cites none. Each
/// warning goes on its own line under its step. A blank summary, and a source list with
/// nothing in it, are left out. An answer that cannot be given is a single line,
/// `Unable to answer: <reason>`.
///
/// Texts are shown as written, apart from their edges: no blank lines or spaces at the
/// start or the end, no spaces at the end of a line and no `\r`. Everything under a step
/// is indented by three spaces, the way `render_chunks` indents the text of a result.
pub fn render_answer(answer: &Answer) -> String {
    match answer {
        Answer::Unable { reason } => format!("Unable to answer: {}", clean_text(reason)),
        Answer::Steps {
            summary,
            steps,
            sources,
        } => {
            let sources = if sources.is_empty() {
                String::new()
            } else {
                render_sources(sources)
            };
            [clean_text(summary), render_steps(steps), sources]
                .into_iter()
                .filter(|block| !block.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        }
    }
}

fn render_steps(steps: &[AnswerStep]) -> String {
    steps
        .iter()
        .zip(1..)
        .map(|(step, number)| render_step(number, step))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_step(number: usize, step: &AnswerStep) -> String {
    let citations = if step.sources.is_empty() {
        "no source".to_owned()
    } else {
        step.sources
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    // The citations are added before the text is split into lines, so they end its last
    // line. The instruction is trimmed first: a trailing newline would leave them alone on
    // a line of their own.
    let text = format!("{} [{citations}]", step.instruction.trim());
    let lines = clean_lines(&text).enumerate().map(|(position, line)| {
        if position == 0 {
            format!("{number}. {line}")
        } else {
            indent(line)
        }
    });
    let warnings = step
        .warnings
        .iter()
        .flat_map(|warning| clean_lines(warning).map(indent));
    lines.chain(warnings).collect::<Vec<_>>().join("\n")
}

fn render_sources(sources: &[Source]) -> String {
    once("Sources:".to_owned())
        .chain(
            sources
                .iter()
                .zip(1..)
                .map(|(source, number)| render_source(number, source)),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_source(number: usize, source: &Source) -> String {
    format!("[{number}] {}, page {}", source.file.display(), source.page)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use super::*;
    use crate::answer::{AnswerStep, Source};

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

    // A blank line in the middle of the text stays, but empty: no line is left holding only
    // spaces.
    #[test]
    fn a_blank_line_inside_the_text_stays_empty() {
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

    // ---- render_answer ----

    fn step(instruction: &str, warnings: &[&str], sources: &[usize]) -> AnswerStep {
        AnswerStep {
            instruction: instruction.to_string(),
            warnings: warnings.iter().map(|w| w.to_string()).collect(),
            sources: sources.to_vec(),
        }
    }

    fn source(file: &str, page: usize) -> Source {
        Source {
            file: PathBuf::from(file),
            page,
        }
    }

    fn procedure(summary: &str, steps: Vec<AnswerStep>, sources: Vec<Source>) -> Answer {
        Answer::Steps {
            summary: summary.to_string(),
            steps,
            sources,
        }
    }

    fn decline(reason: &str) -> Answer {
        Answer::Unable {
            reason: reason.to_string(),
        }
    }

    #[test]
    fn a_procedure_is_the_summary_then_the_numbered_steps_then_the_sources() {
        let answer = procedure(
            "Drain the boiler.",
            vec![
                step("Open the drain valve.", &["WARNING: scalding water."], &[1]),
                step("Close the drain valve again.", &[], &[1, 2]),
                step("Check the seal.", &[], &[2]),
            ],
            vec![source("boiler.pdf", 3), source("pump.pdf", 1)],
        );
        let expected = [
            "Drain the boiler.",
            "",
            "1. Open the drain valve. [1]",
            "   WARNING: scalding water.",
            "2. Close the drain valve again. [1, 2]",
            "3. Check the seal. [2]",
            "",
            "Sources:",
            "[1] boiler.pdf, page 3",
            "[2] pump.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // The number in front of a step is its place in the list. The numbers in brackets are
    // sources, and they can come in any order.
    #[test]
    fn steps_are_numbered_by_position_not_by_the_sources_they_cite() {
        let answer = procedure(
            "",
            vec![
                step("First.", &[], &[3]),
                step("Second.", &[], &[1]),
                step("Third.", &[], &[2]),
            ],
            vec![source("a.pdf", 1), source("b.pdf", 2), source("c.pdf", 3)],
        );
        let expected = [
            "1. First. [3]",
            "2. Second. [1]",
            "3. Third. [2]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
            "[2] b.pdf, page 2",
            "[3] c.pdf, page 3",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    #[test]
    fn a_step_lists_every_source_it_cites() {
        let answer = procedure(
            "",
            vec![step("Do it.", &[], &[1, 2, 3])],
            vec![source("a.pdf", 1), source("b.pdf", 2), source("c.pdf", 3)],
        );

        let shown = render_answer(&answer);

        assert_eq!(shown.lines().next().unwrap(), "1. Do it. [1, 2, 3]");
    }

    // The list keeps the order of the answer. The path is shown as stored, folders
    // included, because two manuals can share a file name.
    #[test]
    fn each_source_is_a_numbered_line_with_its_folders_and_page() {
        let answer = procedure(
            "",
            vec![step("Do it.", &[], &[1, 2])],
            vec![source("boilers/x200/manual.pdf", 12), source("a.pdf", 1)],
        );
        let expected = [
            "1. Do it. [1, 2]",
            "",
            "Sources:",
            "[1] boilers/x200/manual.pdf, page 12",
            "[2] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    #[test]
    fn warnings_sit_under_their_step_each_on_its_own_line_in_order() {
        let answer = procedure(
            "",
            vec![
                step(
                    "Open the valve.",
                    &["WARNING: hot.", "CAUTION: wear gloves."],
                    &[1],
                ),
                step("Close the valve.", &[], &[1]),
                step("Check the seal.", &["DANGER: pressurised."], &[1]),
            ],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve. [1]",
            "   WARNING: hot.",
            "   CAUTION: wear gloves.",
            "2. Close the valve. [1]",
            "3. Check the seal. [1]",
            "   DANGER: pressurised.",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // A step the model could not back up is still part of the procedure, and so are its
    // warnings. The flag is what tells the reader to check it against the manual.
    #[test]
    fn a_step_without_a_source_is_flagged_and_keeps_its_warnings() {
        let answer = procedure(
            "",
            vec![
                step("Open the valve.", &[], &[1]),
                step("Tighten the bolts.", &["WARNING: do not overtighten."], &[]),
                step("Close the valve.", &[], &[1]),
            ],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve. [1]",
            "2. Tighten the bolts. [no source]",
            "   WARNING: do not overtighten.",
            "3. Close the valve. [1]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // With nothing cited there is nothing to list, and a heading with no lines under it
    // would only look like an error.
    #[test]
    fn no_source_list_when_nothing_is_cited() {
        let answer = procedure(
            "Drain the boiler.",
            vec![
                step("Open the valve.", &[], &[]),
                step("Close it.", &[], &[]),
            ],
            vec![],
        );
        let expected = [
            "Drain the boiler.",
            "",
            "1. Open the valve. [no source]",
            "2. Close it. [no source]",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // An empty summary must not leave a blank line at the top of the output.
    #[rstest]
    #[case::empty("")]
    #[case::blank("  \n ")]
    fn without_a_summary_the_output_starts_with_the_first_step(#[case] summary: &str) {
        let answer = procedure(
            summary,
            vec![step("Open the valve.", &[], &[1])],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve. [1]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    #[test]
    fn with_no_summary_and_no_sources_the_output_is_just_the_steps() {
        let answer = procedure(
            "",
            vec![
                step("Open the valve.", &[], &[]),
                step("Close it.", &[], &[]),
            ],
            vec![],
        );

        assert_eq!(
            render_answer(&answer),
            "1. Open the valve. [no source]\n2. Close it. [no source]"
        );
    }

    #[test]
    fn a_summary_is_shown_trimmed_and_not_indented() {
        let answer = procedure(
            "\n  Drain the boiler.\nIt takes ten minutes.  \n",
            vec![step("Open the valve.", &[], &[1])],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "Drain the boiler.",
            "It takes ten minutes.",
            "",
            "1. Open the valve. [1]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // A blank warning would print an empty indented line that looks like a missing one.
    #[test]
    fn a_blank_warning_prints_nothing() {
        let answer = procedure(
            "",
            vec![step(
                "Open the valve.",
                &["", "  \n", "WARNING: hot."],
                &[1],
            )],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve. [1]",
            "   WARNING: hot.",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // The lines after the first stay under their step. The sources close the last line of
    // the instruction, and the warnings come after that line.
    #[test]
    fn every_line_of_a_multi_line_instruction_stays_under_its_number() {
        let answer = procedure(
            "",
            vec![
                step(
                    "Open the valve.\nWait for the gauge to read zero.",
                    &["WARNING: hot."],
                    &[1],
                ),
                step("Close the valve.\nCheck the seal.", &[], &[]),
            ],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve.",
            "   Wait for the gauge to read zero. [1]",
            "   WARNING: hot.",
            "2. Close the valve.",
            "   Check the seal. [no source]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // A warning that is cut across lines in the manual must stay whole and visible.
    #[test]
    fn every_line_of_a_multi_line_warning_is_indented() {
        let answer = procedure(
            "",
            vec![step(
                "Open the cover.",
                &["WARNING: disconnect the power\nbefore opening the cover."],
                &[1],
            )],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the cover. [1]",
            "   WARNING: disconnect the power",
            "   before opening the cover.",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // Like in render_chunks, a blank line in the middle of a text stays, but empty.
    #[test]
    fn a_blank_line_inside_an_instruction_stays_empty() {
        let answer = procedure(
            "",
            vec![step("First part.\n\nSecond part.", &[], &[1])],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. First part.",
            "",
            "   Second part. [1]",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // Text from a PDF or a model can end its lines with \r\n, end with a newline or have
    // trailing spaces. None of that may leave a stray \r behind or push the sources onto
    // a line of their own.
    #[rstest]
    #[case::lf("first\nsecond")]
    #[case::crlf("first\r\nsecond")]
    #[case::trailing_newline("first\nsecond\n")]
    #[case::trailing_spaces("first  \nsecond  ")]
    fn line_endings_and_trailing_spaces_in_an_instruction_make_no_difference(
        #[case] instruction: &str,
    ) {
        let answer = procedure(
            "",
            vec![step(instruction, &[], &[1])],
            vec![source("a.pdf", 1)],
        );

        assert_eq!(
            render_answer(&answer),
            "1. first\n   second [1]\n\nSources:\n[1] a.pdf, page 1"
        );
    }

    #[test]
    fn windows_line_endings_in_a_warning_leave_no_stray_characters() {
        let answer = procedure(
            "",
            vec![step(
                "Open the valve.",
                &["WARNING: hot.\r\nDo not touch.\r\n"],
                &[1],
            )],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "1. Open the valve. [1]",
            "   WARNING: hot.",
            "   Do not touch.",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    // The reader acts on these words, so nothing in them is escaped, converted or cut.
    #[test]
    fn the_text_of_steps_and_warnings_is_shown_as_written() {
        let answer = procedure(
            "Keep the pressure < 2 bar & > 0.5 bar.",
            vec![step(
                "Open the valve to 30\u{b0}C & keep < 2 bar {max} at 100% \"slowly\".",
                &[
                    "WARNING: do not exceed 2 bar & 100\u{b0}C.",
                    "CAUTION: Wear gloves.",
                ],
                &[1],
            )],
            vec![source("a.pdf", 1)],
        );
        let expected = [
            "Keep the pressure < 2 bar & > 0.5 bar.",
            "",
            "1. Open the valve to 30\u{b0}C & keep < 2 bar {max} at 100% \"slowly\". [1]",
            "   WARNING: do not exceed 2 bar & 100\u{b0}C.",
            "   CAUTION: Wear gloves.",
            "",
            "Sources:",
            "[1] a.pdf, page 1",
        ]
        .join("\n");

        assert_eq!(render_answer(&answer), expected);
    }

    #[rstest]
    #[case::plain("The passages do not cover the pump.")]
    #[case::padded("  The passages do not cover the pump.  \r\n")]
    fn a_decline_is_one_line_with_the_reason(#[case] reason: &str) {
        assert_eq!(
            render_answer(&decline(reason)),
            "Unable to answer: The passages do not cover the pump."
        );
    }

    fn messy_procedure() -> Answer {
        procedure(
            "  Drain the boiler.  \r\n",
            vec![
                step(
                    "Open the valve.  \r\n\r\n",
                    &["WARNING: hot.   \n", " \n"],
                    &[1],
                ),
                step("Wait.   \nThen close it.  ", &[], &[]),
            ],
            vec![source("boiler.pdf", 3)],
        )
    }

    // println! adds its own newline, and spaces at the end of a line are invisible but
    // show up when the output is copied or compared.
    #[rstest]
    #[case::procedure(messy_procedure())]
    #[case::decline(decline("  Nothing about the pump.  \r\n"))]
    fn messy_whitespace_never_reaches_the_screen(#[case] answer: Answer) {
        let shown = render_answer(&answer);

        assert!(!shown.contains('\r'), "got: {shown:?}");
        assert!(!shown.ends_with('\n'), "got: {shown:?}");
        for line in shown.lines() {
            assert_eq!(line, line.trim_end(), "got: {shown:?}");
        }
    }
}
