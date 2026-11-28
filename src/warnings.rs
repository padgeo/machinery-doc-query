// The model is asked to copy every WARNING, CAUTION and DANGER of the passages it cites into
// the step they apply to. A warning it leaves out is the worst mistake it can make, so the
// program checks the copying itself: this is how the warnings of a passage are found, and how
// a step is seen to carry one.

// The words that start a warning in a manual.
const WARNING_LABELS: [&str; 3] = ["WARNING", "CAUTION", "DANGER"];
// The marks that start an item of a bulleted list.
const BULLETS: [&str; 3] = ["-", "*", "•"];

/// The warnings in the text of a passage, each as one line of text, in the order they stand.
///
/// A warning starts at a line that begins with WARNING, CAUTION or DANGER as a whole word,
/// either written in capitals (`WARNING`, `DANGER!`) or followed by a colon (`Warning:`). It
/// runs to the end of its paragraph: it stops at a blank line, at the next warning and at a
/// line that starts a list item (a number followed by a dot or a bracket, or `-`, `*` or `•`,
/// and then a space). Its lines are joined with single spaces, and nothing else in its text is
/// changed.
pub fn find_warnings(text: &str) -> Vec<String> {
    // One entry per warning, holding the lines that belong to it. They are joined into one line
    // at the end, when it is known where each warning stops.
    let mut warnings: Vec<Vec<&str>> = Vec::new();
    // Whether the last warning still takes the lines that follow it. It is open from its label
    // until a blank line, a list item or the next warning ends it.
    let mut open = false;
    // Lines are trimmed first: a PDF indents them, and a Windows file ends them with \r\n.
    // A line that belongs to no warning matches none of the branches and is skipped.
    for line in text.lines().map(str::trim) {
        if starts_a_warning(line) {
            // A label starts a new warning. That ends the one before it, because the new one
            // goes last and only the last can be open.
            warnings.push(vec![line]);
            open = true;
        } else if line.is_empty() || starts_a_list_item(line) {
            // A blank line or a list item ends the warning: what follows is not part of it.
            open = false;
        } else if open && let Some(last) = warnings.last_mut() {
            // Any other line right after a warning continues it. A PDF cuts its lines where the
            // page ends, not where the sentence does, and a label can stand alone on a line.
            last.push(line);
        }
    }
    // A warning that ran over several lines becomes one line of text.
    warnings.iter().map(|lines| lines.join(" ")).collect()
}

/// Whether `line` starts a warning: it begins with one of the labels, as a whole word, and the
/// label is either in capitals or followed by a colon.
fn starts_a_warning(line: &str) -> bool {
    WARNING_LABELS.iter().any(|label| {
        // `get` gives None, and not a panic, when the length of the label falls inside a
        // character that takes several bytes. Such a line cannot start with an ASCII label.
        let Some(head) = line.get(..label.len()) else {
            return false;
        };

        // The line has to start with the label, in whatever case.
        if !head.eq_ignore_ascii_case(label) {
            return false;
        }

        // A whole word: "WARNINGS" and "Cautionary" only start like a label. Slicing is safe,
        // because `get` has just shown that `label.len()` is a character boundary.
        let rest = &line[label.len()..];
        if rest.chars().next().is_some_and(char::is_alphanumeric) {
            return false;
        }
        // Capitals ("WARNING Hot", "DANGER!") are a label whatever follows. In any other case a
        // colon is needed ("Warning: hot"), or it may be a sentence ("Warning signs include...").
        head == *label || rest.starts_with(':')
    })
}

/// Whether `line` starts a list item: a number followed by a dot or a bracket, or a bullet, and
/// then a space.
fn starts_a_list_item(line: &str) -> bool {
    // The digits at the start of the line. They take one byte each, so their count is also where
    // the rest of the line begins.
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    let after_the_number = &line[digits..];
    // "3. Open" and "3) Open" are items. The space matters: "3.5 mm" is a measure, and the
    // warning goes on.
    let numbered =
        digits > 0 && (after_the_number.starts_with(". ") || after_the_number.starts_with(") "));
    // "- Open", "* Open" and "• Open" are items. Here too the space matters: "-5 degrees" is not.
    let bulleted = BULLETS.iter().any(|bullet| {
        line.strip_prefix(bullet)
            .is_some_and(|rest| rest.starts_with(' '))
    });
    numbered || bulleted
}

/// Whether `warning`, as a step has it, carries `found`, a warning found in a passage with
/// `find_warnings`: it has the same words, whatever the case, the spacing between the words or
/// the full stop at the end of `found`. Other text around them does not matter, so a step can
/// hold several warnings in one.
pub fn carries(warning: &str, found: &str) -> bool {
    let found = words(found);
    words(warning).contains(found.trim_end_matches('.'))
}

/// The words of a text, in lower case and separated by single spaces, so that two texts that
/// differ only in case or in spacing come out equal.
fn words(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    // ---- what a warning is ----

    #[rstest]
    #[case::warning("WARNING: Allow the unit to cool for 30 minutes.")]
    #[case::caution("CAUTION: Do not use abrasive cleaners.")]
    #[case::danger("DANGER: Live wires behind the cover.")]
    fn a_line_with_one_of_the_three_labels_is_a_warning(#[case] line: &str) {
        assert_eq!(find_warnings(line), vec![line]);
    }

    // Manuals write the label in capitals, with or without a colon, or as a heading with a colon.
    #[rstest]
    #[case::capitals_without_a_colon("WARNING Hot surface.")]
    #[case::capitals_and_an_exclamation_mark("DANGER! Live wires.")]
    #[case::capitals_and_a_dash("CAUTION - Heavy part.")]
    #[case::capitalised_with_a_colon("Warning: hot surface.")]
    #[case::lower_case_with_a_colon("caution: heavy part.")]
    fn the_ways_a_manual_writes_the_label_are_found(#[case] line: &str) {
        assert_eq!(find_warnings(line), vec![line]);
    }

    // A word that only starts like a label, a label that is no heading, and a label that is not
    // at the start of its line are not warnings.
    #[rstest]
    #[case::not_in_capitals_and_no_colon("Warning signs include smoke.")]
    #[case::a_longer_word("WARNINGS: read first.")]
    #[case::a_longer_word_in_capitals("DANGEROUS goods.")]
    #[case::a_longer_word_with_a_colon("Cautionary note: wipe it.")]
    #[case::another_label("NOTE: handle with care.")]
    #[case::not_at_the_start("Keep clear. WARNING: hot.")]
    fn text_that_is_not_a_warning_is_left_alone(#[case] line: &str) {
        assert!(find_warnings(line).is_empty());
    }

    #[test]
    fn the_warnings_of_a_text_are_found_among_its_steps_in_order() {
        let text = [
            "1. Turn off the water.",
            "WARNING: Scalding water.",
            "2. Open the valve.",
            "",
            "CAUTION: Heavy part.",
            "3. Lift it.",
        ]
        .join("\n");

        assert_eq!(
            find_warnings(&text),
            vec!["WARNING: Scalding water.", "CAUTION: Heavy part."]
        );
    }

    #[rstest]
    #[case::empty("")]
    #[case::blank("  \n \n")]
    #[case::steps_only("1. Open the valve.\n2. Close it.")]
    fn a_text_without_warnings_gives_none(#[case] text: &str) {
        assert!(find_warnings(text).is_empty());
    }

    // ---- where a warning ends ----

    // A line of a PDF is cut where the page ends, not where the sentence does.
    #[test]
    fn a_warning_that_runs_over_several_lines_is_one_warning() {
        let text = "WARNING: Disconnect the power supply before\nremoving the cover.";

        assert_eq!(
            find_warnings(text),
            vec!["WARNING: Disconnect the power supply before removing the cover."]
        );
    }

    // A warning can be a heading with its text under it.
    #[test]
    fn a_label_on_a_line_of_its_own_takes_the_lines_under_it() {
        let text = "WARNING\nDisconnect the power supply.\n\nOpen the cover.";

        assert_eq!(
            find_warnings(text),
            vec!["WARNING Disconnect the power supply."]
        );
    }

    #[rstest]
    #[case::a_blank_line("WARNING: Hot.\n\nThis is another paragraph.")]
    #[case::a_numbered_step("WARNING: Hot.\n3. Open the cover.")]
    #[case::a_bracketed_number("WARNING: Hot.\n3) Open the cover.")]
    #[case::a_dash("WARNING: Hot.\n- Open the cover.")]
    #[case::a_star("WARNING: Hot.\n* Open the cover.")]
    #[case::a_bullet("WARNING: Hot.\n\u{2022} Open the cover.")]
    fn a_warning_ends_where_its_paragraph_or_the_next_list_item_starts(#[case] text: &str) {
        assert_eq!(find_warnings(text), vec!["WARNING: Hot."]);
    }

    #[test]
    fn the_next_warning_ends_a_warning() {
        let text = "WARNING: Hot.\nCAUTION: Heavy.";

        assert_eq!(
            find_warnings(text),
            vec!["WARNING: Hot.", "CAUTION: Heavy."]
        );
    }

    // A number or a minus sign in the middle of a sentence is not a list.
    #[test]
    fn a_line_that_starts_with_a_number_but_is_no_list_item_goes_on_with_the_warning() {
        let text = "WARNING: Keep the gap to\n3.5 mm\n-5 degrees at most.";

        assert_eq!(
            find_warnings(text),
            vec!["WARNING: Keep the gap to 3.5 mm -5 degrees at most."]
        );
    }

    // ---- the text of a warning ----

    #[test]
    fn indentation_and_windows_line_endings_make_no_difference() {
        let text = "  WARNING: Hot\r\n    surface.\r\n";

        assert_eq!(find_warnings(text), vec!["WARNING: Hot surface."]);
    }

    // The words are the manual's own: the case, and the spacing inside a line, stay.
    #[test]
    fn the_text_of_a_warning_is_kept_as_written() {
        let text = "Warning:  keep  below 2 bar & 80\u{b0}C";

        assert_eq!(find_warnings(text), vec![text]);
    }

    // The label is looked for in the first bytes of a line, and those must not be cut inside a
    // character.
    #[test]
    fn text_in_other_alphabets_is_no_problem() {
        let text = "\u{c9}\u{c9}\u{c9}\u{c9}\u{c9}\u{c9}\n\nWARNING: \u{fc}ber 80\u{b0}C\n\n\u{c9}CAUTION: no";

        assert_eq!(find_warnings(text), vec!["WARNING: \u{fc}ber 80\u{b0}C"]);
    }

    // ---- carries ----

    const FOUND: &str = "WARNING: Allow the unit to cool for 30 minutes.";

    #[rstest]
    #[case::a_copy(FOUND)]
    #[case::another_case("warning: ALLOW the unit to cool for 30 MINUTES.")]
    #[case::other_spacing("WARNING:   Allow the unit\n to cool  for 30 minutes.")]
    #[case::without_the_last_full_stop("WARNING: Allow the unit to cool for 30 minutes")]
    #[case::with_text_around_it(
        "Read this first. WARNING: Allow the unit to cool for 30 minutes. Then drain it."
    )]
    fn a_step_carries_a_warning_it_copied(#[case] warning: &str) {
        assert!(carries(warning, FOUND));
    }

    #[test]
    fn one_warning_of_a_step_can_hold_several() {
        let warning = "WARNING: Hot. CAUTION: Heavy.";

        assert!(carries(warning, "WARNING: Hot."));
        assert!(carries(warning, "CAUTION: Heavy."));
    }

    #[rstest]
    #[case::a_paraphrase("WARNING: Let the unit cool down first.")]
    #[case::cut_short("WARNING: Allow the unit to cool for 30")]
    #[case::another_warning("WARNING: Allow the unit to dry for 30 minutes.")]
    #[case::nothing("")]
    fn a_step_does_not_carry_a_warning_it_changed_or_cut(#[case] warning: &str) {
        assert!(!carries(warning, FOUND));
    }
}
