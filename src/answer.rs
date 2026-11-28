use std::path::PathBuf;

use crate::{
    ai::schema::StructuredAnswer,
    chunk::RetrievedChunk,
    warnings::{carries, find_warnings},
};

/// A page of a document that an answer cites.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub file: PathBuf,
    pub page: usize,
}

/// One step of an answer, after its citations were checked against the passages that
/// were sent to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerStep {
    pub instruction: String,
    pub warnings: Vec<String>,
    /// The sources this step cites, as numbers from 1 into the `sources` of the answer,
    /// in ascending order. Empty when none of the numbers the model gave pointed at a
    /// passage.
    pub sources: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// A procedure. `sources` holds only the pages that were cited, numbered in the
    /// order they are first cited.
    Steps {
        summary: String,
        steps: Vec<AnswerStep>,
        sources: Vec<Source>,
    },
    /// There is no procedure to show. `reason` is meant for the user.
    Unable { reason: String },
}

const FALLBACK_REASON: &str = "No reason given.";
const NO_STEPS_REASON: &str = "No steps given.";

/// Checks and groups the citations for the answer.
/// Filters out duplicate citations and dedups source numbers per step and the pages in the source list.
/// Source numbers pointing nowhere are dropped.
/// Returns Unable answer if the AI returned that it was unable to answer
/// or no steps were given in the answer.
pub fn check_citations(answer: StructuredAnswer, passages: &[RetrievedChunk]) -> Answer {
    if answer.unable_to_answer {
        let trimmed = answer.summary.trim();
        let reason = if trimmed.is_empty() {
            FALLBACK_REASON.to_owned()
        } else {
            trimmed.to_owned()
        };
        return Answer::Unable { reason };
    }

    if answer.steps.is_empty() {
        return Answer::Unable {
            reason: NO_STEPS_REASON.to_string(),
        };
    }

    let mut sources: Vec<Source> = Vec::new();
    let mut steps = Vec::new();
    // For each step, the passages it cites (by position, each once), in the order of `steps`.
    let mut cited = Vec::new();

    for step in answer.steps {
        let mut labels = Vec::new();
        let mut positions = Vec::new();
        for number in step.source_indices {
            let Some(passage) = passage_for(number, passages) else {
                continue;
            };
            let passage_chunk = &passages[passage];
            let source = Source {
                file: passage_chunk.chunk.source_file.clone(),
                page: passage_chunk.chunk.page_number,
            };
            let label = match sources.iter().position(|known| *known == source) {
                Some(position) => position + 1,
                None => {
                    sources.push(source);
                    sources.len()
                }
            };
            labels.push(label);
            // Several numbers can lead to the same passage. It counts once.
            if !positions.contains(&passage) {
                positions.push(passage);
            }
        }
        labels.sort_unstable();
        labels.dedup();
        cited.push(positions);
        steps.push(AnswerStep {
            instruction: step.instruction,
            warnings: step.warnings,
            sources: labels,
        });
    }
    // The citations are settled, so the warnings of the cited passages can be checked.
    add_missing_warnings(&mut steps, &cited, passages);
    Answer::Steps {
        summary: answer.summary,
        steps,
        sources,
    }
}

fn passage_for(number: i64, passages: &[RetrievedChunk]) -> Option<usize> {
    let number = usize::try_from(number).ok()?; // negative -> None
    let position = number.checked_sub(1)?; // 0 -> None
    (position < passages.len()).then_some(position) // too big -> None
}

/// Every warning of a cited passage has to be under a step that cites the passage. When none
/// of those steps carries it, it goes to the first of them.
fn add_missing_warnings(
    steps: &mut [AnswerStep],
    cited: &[Vec<usize>],
    passages: &[RetrievedChunk],
) {
    // The passages that are cited, each once, in the order they are first cited. That is the
    // order of the sources, and it decides the order the added warnings come out in.
    let mut in_order_of_citation: Vec<usize> = Vec::new();
    for &position in cited.iter().flatten() {
        if !in_order_of_citation.contains(&position) {
            in_order_of_citation.push(position);
        }
    }

    for position in in_order_of_citation {
        // The steps that cite this passage, as indexes into `steps`. Only they can carry its
        // warnings, and the first of them gets the ones that are missing. The list is never
        // empty: the passage is here because some step cites it.
        let citing: Vec<usize> = (0..steps.len())
            .filter(|&step| cited[step].contains(&position))
            .collect();
        for found in find_warnings(&passages[position].chunk.text) {
            // Does one of those steps already carry this warning? The warnings are read as they
            // are now, so one that was just added counts, and a warning that stands in two
            // overlapping passages is added once. Indexes and not references keep the borrow
            // checker quiet: this read is over before the `push` below changes `steps`.
            let carried = citing.iter().any(|&step| {
                steps[step]
                    .warnings
                    .iter()
                    .any(|warning| carries(warning, &found))
            });
            if !carried {
                steps[citing[0]].warnings.push(found);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{ai::schema::Step as ModelStep, chunk::Chunk};

    fn passage(file: &str, page: usize) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk {
                source_file: PathBuf::from(file),
                page_number: page,
                chunk_index: 0,
                text: "some text".to_string(),
            },
            score: 0.5,
        }
    }

    // Passage 1 is a.pdf page 1, passage 2 is b.pdf page 2, passage 3 is c.pdf page 3.
    fn three_passages() -> Vec<RetrievedChunk> {
        vec![
            passage("a.pdf", 1),
            passage("b.pdf", 2),
            passage("c.pdf", 3),
        ]
    }

    fn source(file: &str, page: usize) -> Source {
        Source {
            file: PathBuf::from(file),
            page,
        }
    }

    // A step as the model wrote it: the passage numbers are whatever it said.
    fn model_step(instruction: &str, warnings: &[&str], numbers: &[i64]) -> ModelStep {
        ModelStep {
            instruction: instruction.to_string(),
            warnings: warnings.iter().map(|w| w.to_string()).collect(),
            source_indices: numbers.to_vec(),
        }
    }

    fn model_answer(steps: Vec<ModelStep>) -> StructuredAnswer {
        StructuredAnswer {
            summary: "Drain the boiler.".to_string(),
            unable_to_answer: false,
            steps,
        }
    }

    // Runs the check on an answer made of these steps, and expects a procedure back.
    fn checked(
        steps: Vec<ModelStep>,
        passages: &[RetrievedChunk],
    ) -> (Vec<AnswerStep>, Vec<Source>) {
        match check_citations(model_answer(steps), passages) {
            Answer::Steps { steps, sources, .. } => (steps, sources),
            other => panic!("expected a procedure, got {other:?}"),
        }
    }

    // ---- which passage a number points at ----

    #[rstest]
    #[case::the_first(1, "a.pdf", 1)]
    #[case::the_second(2, "b.pdf", 2)]
    #[case::the_third(3, "c.pdf", 3)]
    fn a_number_points_at_the_passage_in_that_position_counting_from_one(
        #[case] number: i64,
        #[case] file: &str,
        #[case] page: usize,
    ) {
        let steps = vec![model_step("Do it.", &[], &[number])];

        let (steps, sources) = checked(steps, &three_passages());

        assert_eq!(sources, vec![source(file, page)]);
        assert_eq!(steps[0].sources, vec![1]);
    }

    // The model can write any number at all. None of these may panic or point anywhere.
    #[rstest]
    #[case::zero(0)]
    #[case::negative(-1)]
    #[case::one_past_the_end(4)]
    #[case::huge(i64::MAX)]
    #[case::most_negative(i64::MIN)]
    fn a_number_that_points_at_no_passage_is_dropped(#[case] number: i64) {
        let steps = vec![model_step("Do it.", &[], &[number])];

        let (steps, sources) = checked(steps, &three_passages());

        assert!(steps[0].sources.is_empty());
        assert!(sources.is_empty());
    }

    #[test]
    fn a_bad_number_does_not_take_the_good_ones_with_it() {
        let steps = vec![model_step("Do it.", &[], &[0, 2, 99])];

        let (steps, sources) = checked(steps, &three_passages());

        assert_eq!(sources, vec![source("b.pdf", 2)]);
        assert_eq!(steps[0].sources, vec![1]);
    }

    #[test]
    fn the_same_passage_cited_twice_in_a_step_counts_once() {
        let steps = vec![model_step("Do it.", &[], &[2, 2, 2])];

        let (steps, sources) = checked(steps, &three_passages());

        assert_eq!(steps[0].sources, vec![1]);
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn with_no_passages_every_number_is_dropped_and_nothing_panics() {
        let steps = vec![model_step("Do it.", &[], &[1])];

        let (steps, sources) = checked(steps, &[]);

        assert!(steps[0].sources.is_empty());
        assert!(sources.is_empty());
    }

    // ---- a source is a page, not a passage ----

    // A page is cut into several passages, so passages next to each other often share a
    // page. Citing both is one source.
    #[test]
    fn passages_from_the_same_page_are_one_source() {
        let passages = [
            passage("a.pdf", 3),
            passage("a.pdf", 3),
            passage("b.pdf", 1),
        ];
        let steps = vec![model_step("Do it.", &[], &[1, 2, 3])];

        let (steps, sources) = checked(steps, &passages);

        assert_eq!(sources, vec![source("a.pdf", 3), source("b.pdf", 1)]);
        assert_eq!(steps[0].sources, vec![1, 2]);
    }

    #[test]
    fn the_same_file_on_two_pages_is_two_sources() {
        let passages = [passage("a.pdf", 3), passage("a.pdf", 4)];
        let steps = vec![model_step("Do it.", &[], &[1, 2])];

        let (steps, sources) = checked(steps, &passages);

        assert_eq!(sources, vec![source("a.pdf", 3), source("a.pdf", 4)]);
        assert_eq!(steps[0].sources, vec![1, 2]);
    }

    #[test]
    fn the_same_page_number_in_two_files_is_two_sources() {
        let passages = [passage("a.pdf", 3), passage("b.pdf", 3)];
        let steps = vec![model_step("Do it.", &[], &[1, 2])];

        let (steps, sources) = checked(steps, &passages);

        assert_eq!(sources, vec![source("a.pdf", 3), source("b.pdf", 3)]);
        assert_eq!(steps[0].sources, vec![1, 2]);
    }

    // Two versions of a manual often share a file name and differ only by folder.
    #[test]
    fn the_same_file_name_in_two_folders_is_two_sources() {
        let passages = [passage("old/manual.pdf", 3), passage("new/manual.pdf", 3)];
        let steps = vec![model_step("Do it.", &[], &[1, 2])];

        let (steps, sources) = checked(steps, &passages);

        assert_eq!(
            sources,
            vec![source("old/manual.pdf", 3), source("new/manual.pdf", 3)]
        );
        assert_eq!(steps[0].sources, vec![1, 2]);
    }

    // ---- the numbering of the sources ----

    #[test]
    fn sources_are_numbered_in_the_order_they_are_first_cited() {
        let steps = vec![
            model_step("One.", &[], &[3]),
            model_step("Two.", &[], &[1, 3]),
            model_step("Three.", &[], &[2]),
        ];

        let (steps, sources) = checked(steps, &three_passages());

        assert_eq!(
            sources,
            vec![source("c.pdf", 3), source("a.pdf", 1), source("b.pdf", 2)]
        );
        assert_eq!(steps[0].sources, vec![1]);
        // passage 1 is source 2 and passage 3 is source 1, and a step lists them ascending
        assert_eq!(steps[1].sources, vec![1, 2]);
        assert_eq!(steps[2].sources, vec![3]);
    }

    #[test]
    fn two_new_sources_in_one_step_are_numbered_in_the_order_the_model_wrote_them() {
        let steps = vec![model_step("One.", &[], &[3, 1])];

        let (steps, sources) = checked(steps, &three_passages());

        assert_eq!(sources, vec![source("c.pdf", 3), source("a.pdf", 1)]);
        assert_eq!(steps[0].sources, vec![1, 2]);
    }

    #[test]
    fn only_the_cited_passages_appear_in_the_sources() {
        let steps = vec![model_step("One.", &[], &[2])];

        let (_, sources) = checked(steps, &three_passages());

        assert_eq!(sources, vec![source("b.pdf", 2)]);
    }

    // ---- steps, and what is left of them ----

    // A step the model could not back up is still part of the procedure. Dropping it would
    // hide a step. It is kept, and having no sources is what marks it.
    #[test]
    fn a_step_left_without_any_valid_source_is_kept_and_has_none() {
        let steps = vec![
            model_step("One.", &["WARNING: hot."], &[1]),
            model_step("Two.", &["CAUTION: heavy."], &[0, 42]),
            model_step("Three.", &[], &[]),
        ];

        let (steps, _) = checked(steps, &three_passages());

        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].sources, vec![1]);
        assert_eq!(steps[1].instruction, "Two.");
        assert_eq!(steps[1].warnings, vec!["CAUTION: heavy."]);
        assert!(steps[1].sources.is_empty());
        assert!(steps[2].sources.is_empty());
    }

    // A warning must reach the reader as the manual wrote it.
    #[test]
    fn the_text_of_steps_and_warnings_is_passed_on_exactly_and_in_order() {
        let steps = vec![
            model_step(
                "Open the valve to 30\u{b0}C.",
                &["WARNING: keep < 2 bar & > 0.5 bar", "CAUTION: Wear gloves."],
                &[1],
            ),
            model_step("  Close it again.  ", &[], &[2]),
        ];

        let (steps, _) = checked(steps, &three_passages());

        assert_eq!(steps[0].instruction, "Open the valve to 30\u{b0}C.");
        assert_eq!(
            steps[0].warnings,
            vec!["WARNING: keep < 2 bar & > 0.5 bar", "CAUTION: Wear gloves."]
        );
        assert_eq!(steps[1].instruction, "  Close it again.  ");
    }

    #[test]
    fn the_summary_is_kept() {
        let answer = model_answer(vec![model_step("One.", &[], &[1])]);

        match check_citations(answer, &three_passages()) {
            Answer::Steps { summary, .. } => assert_eq!(summary, "Drain the boiler."),
            other => panic!("expected a procedure, got {other:?}"),
        }
    }

    // ---- declining ----

    // Whatever steps came with a refusal were not meant to be followed.
    #[test]
    fn an_answer_that_says_it_cannot_answer_has_no_steps_even_if_the_model_listed_some() {
        let answer = StructuredAnswer {
            summary: "The passages do not cover the pump.".to_string(),
            unable_to_answer: true,
            steps: vec![model_step("Guess.", &[], &[1])],
        };

        let checked = check_citations(answer, &three_passages());

        assert_eq!(
            checked,
            Answer::Unable {
                reason: "The passages do not cover the pump.".to_string()
            }
        );
    }

    // The usual way for the model to decline: the flag, the reason in the summary, no steps.
    #[test]
    fn a_decline_with_no_steps_keeps_the_reason_the_model_gave() {
        let answer = StructuredAnswer {
            summary: "The passages do not cover the pump.".to_string(),
            unable_to_answer: true,
            steps: vec![],
        };

        let checked = check_citations(answer, &three_passages());

        assert_eq!(
            checked,
            Answer::Unable {
                reason: "The passages do not cover the pump.".to_string()
            }
        );
    }

    #[rstest]
    #[case::empty("")]
    #[case::blank("  \n ")]
    fn a_decline_always_has_a_reason_to_show(#[case] summary: &str) {
        let answer = StructuredAnswer {
            summary: summary.to_string(),
            unable_to_answer: true,
            steps: vec![],
        };

        match check_citations(answer, &three_passages()) {
            Answer::Unable { reason } => assert!(!reason.trim().is_empty()),
            other => panic!("expected a decline, got {other:?}"),
        }
    }

    // Steps listed next to a blank reason must not leave the reader with an empty message.
    #[rstest]
    #[case::empty("")]
    #[case::blank("  \n ")]
    fn a_decline_that_lists_steps_still_has_a_reason_to_show(#[case] summary: &str) {
        let answer = StructuredAnswer {
            summary: summary.to_string(),
            unable_to_answer: true,
            steps: vec![model_step("Guess.", &[], &[1])],
        };

        match check_citations(answer, &three_passages()) {
            Answer::Unable { reason } => assert!(!reason.trim().is_empty()),
            other => panic!("expected a decline, got {other:?}"),
        }
    }

    // The model says it is answering but gives no steps, so its summary is prose that no
    // passage was cited for. It must not be shown as if it were the reason.
    #[test]
    fn an_answer_with_no_steps_is_a_decline_and_its_summary_is_not_passed_on() {
        let answer = StructuredAnswer {
            summary: "Just open the valve and wait.".to_string(),
            unable_to_answer: false,
            steps: vec![],
        };

        match check_citations(answer, &three_passages()) {
            Answer::Unable { reason } => {
                assert!(!reason.trim().is_empty());
                assert!(!reason.contains("open the valve"), "got: {reason}");
            }
            other => panic!("expected a decline, got {other:?}"),
        }
    }

    // ---- warnings the model left out ----

    // A passage with a text of its own.
    fn passage_with(file: &str, page: usize, text: &str) -> RetrievedChunk {
        let mut passage = passage(file, page);
        passage.chunk.text = text.to_string();
        passage
    }

    // The model is asked to copy every warning of the passages it cites into its steps. When it
    // forgets one, the warning of the manual is added to the step that cites the passage, so
    // that it cannot get lost.
    #[test]
    fn a_warning_the_model_left_out_is_added_to_the_step_that_cites_its_passage() {
        let passages = [passage_with(
            "a.pdf",
            1,
            "1. Open the valve.\nWARNING: Scalding water.",
        )];
        let steps = vec![model_step("Open the valve.", &[], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Scalding water."]);
    }

    #[test]
    fn a_warning_the_step_carries_is_not_added_again() {
        let passages = [passage_with("a.pdf", 1, "WARNING: Scalding water.")];
        let steps = vec![model_step(
            "Open the valve.",
            &["WARNING: Scalding water."],
            &[1],
        )];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Scalding water."]);
    }

    // A copy that differs only in case, in spacing or in the last full stop is still the same
    // warning, and the step keeps it as it is.
    #[rstest]
    #[case::another_case("warning: SCALDING water.")]
    #[case::other_spacing("WARNING:   Scalding\nwater.")]
    #[case::without_the_last_full_stop("WARNING: Scalding water")]
    #[case::with_more_text("WARNING: Scalding water. Wear gloves.")]
    fn a_copy_that_differs_a_little_is_still_a_copy(#[case] warning: &str) {
        let passages = [passage_with("a.pdf", 1, "WARNING: Scalding water.")];
        let steps = vec![model_step("Open the valve.", &[warning], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec![warning]);
    }

    // A paraphrase is not the wording of the manual, which is what the model was asked for. Both
    // stay: what the model wrote, and then the warning of the manual.
    #[test]
    fn a_paraphrase_does_not_count_so_the_warning_of_the_manual_is_added_after_it() {
        let passages = [passage_with("a.pdf", 1, "WARNING: Scalding water.")];
        let steps = vec![model_step(
            "Open the valve.",
            &["Be careful with the hot water."],
            &[1],
        )];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(
            steps[0].warnings,
            vec!["Be careful with the hot water.", "WARNING: Scalding water."]
        );
    }

    // Only the passages that steps cite are looked at. The model left the others aside.
    #[test]
    fn a_passage_no_step_cites_is_not_looked_at() {
        let passages = [
            passage_with("a.pdf", 1, "Open the valve."),
            passage_with("b.pdf", 2, "WARNING: Hot pipe."),
        ];
        let steps = vec![model_step("Open the valve.", &[], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert!(steps[0].warnings.is_empty());
    }

    #[test]
    fn the_warning_goes_to_the_first_step_that_cites_the_passage() {
        let passages = [passage_with("a.pdf", 1, "WARNING: Hot.")];
        let steps = vec![model_step("One.", &[], &[1]), model_step("Two.", &[], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Hot."]);
        assert!(steps[1].warnings.is_empty());
    }

    // One of the steps that cite the passage carries it, and that is enough.
    #[test]
    fn a_warning_that_any_step_citing_the_passage_carries_is_not_added_to_the_others() {
        let passages = [passage_with("a.pdf", 1, "WARNING: Hot.")];
        let steps = vec![
            model_step("One.", &[], &[1]),
            model_step("Two.", &["WARNING: Hot."], &[1]),
        ];

        let (steps, _) = checked(steps, &passages);

        assert!(steps[0].warnings.is_empty());
        assert_eq!(steps[1].warnings, vec!["WARNING: Hot."]);
    }

    // The step that carries the warning has to be one that cites the passage it is from: a copy
    // under a step that cites another passage cannot be traced to its page.
    #[test]
    fn a_step_that_does_not_cite_the_passage_does_not_count_as_carrying_its_warning() {
        let passages = [
            passage_with("a.pdf", 1, "WARNING: Hot."),
            passage_with("b.pdf", 2, "Close the valve."),
        ];
        let steps = vec![
            model_step("One.", &[], &[1]),
            model_step("Two.", &["WARNING: Hot."], &[2]),
        ];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Hot."]);
        assert_eq!(steps[1].warnings, vec!["WARNING: Hot."]);
    }

    #[test]
    fn added_warnings_follow_the_ones_the_model_gave_in_the_order_of_the_passage() {
        let passages = [passage_with(
            "a.pdf",
            1,
            "WARNING: One.\n1. Open the valve.\nCAUTION: Two.",
        )];
        let steps = vec![model_step("Open the valve.", &["DANGER: Its own."], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(
            steps[0].warnings,
            vec!["DANGER: Its own.", "WARNING: One.", "CAUTION: Two."]
        );
    }

    // The passages are taken in the order they are first cited, the order of the sources.
    #[test]
    fn the_warnings_of_several_passages_are_added_in_the_order_the_passages_are_first_cited() {
        let passages = [
            passage_with("a.pdf", 1, "WARNING: One."),
            passage_with("b.pdf", 2, "WARNING: Two."),
        ];
        let steps = vec![model_step("Do it.", &[], &[2, 1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Two.", "WARNING: One."]);
    }

    // The text of neighbouring passages overlaps, so one warning can stand in both of them.
    #[test]
    fn a_warning_in_two_cited_passages_is_added_once() {
        let passages = [
            passage_with("a.pdf", 1, "1. Open the valve.\nWARNING: Hot."),
            passage_with("a.pdf", 1, "WARNING: Hot.\n2. Close it."),
        ];
        let steps = vec![model_step("Do it.", &[], &[1, 2])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(steps[0].warnings, vec!["WARNING: Hot."]);
    }

    // A step with no valid source cites nothing, so no passage is its own.
    #[test]
    fn a_step_without_a_valid_source_gets_no_warning() {
        let passages = [passage_with("a.pdf", 1, "WARNING: Hot.")];
        let steps = vec![
            model_step("One.", &[], &[0, 9]),
            model_step("Two.", &[], &[1]),
        ];

        let (steps, _) = checked(steps, &passages);

        assert!(steps[0].warnings.is_empty());
        assert_eq!(steps[1].warnings, vec!["WARNING: Hot."]);
    }

    // A warning of two lines is one warning, shown as one line.
    #[test]
    fn a_warning_over_several_lines_is_added_as_one_line() {
        let passages = [passage_with(
            "a.pdf",
            1,
            "WARNING: Disconnect the power\nbefore removing the cover.",
        )];
        let steps = vec![model_step("Remove the cover.", &[], &[1])];

        let (steps, _) = checked(steps, &passages);

        assert_eq!(
            steps[0].warnings,
            vec!["WARNING: Disconnect the power before removing the cover."]
        );
    }

    // The sources, and the numbers that point at them, are the same with and without the warning.
    #[test]
    fn adding_a_warning_changes_neither_the_sources_nor_their_numbers() {
        let with_warning = [
            passage_with("a.pdf", 1, "WARNING: Hot."),
            passage_with("b.pdf", 2, "Close it."),
        ];
        let without = [
            passage_with("a.pdf", 1, "Open it."),
            passage_with("b.pdf", 2, "Close it."),
        ];
        let steps = || {
            vec![
                model_step("One.", &[], &[2, 1]),
                model_step("Two.", &[], &[1]),
            ]
        };

        let (steps_with, sources_with) = checked(steps(), &with_warning);
        let (steps_without, sources_without) = checked(steps(), &without);

        assert_eq!(sources_with, sources_without);
        for (with, without) in steps_with.iter().zip(&steps_without) {
            assert_eq!(with.instruction, without.instruction);
            assert_eq!(with.sources, without.sources);
        }
    }
}
