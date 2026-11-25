use std::fmt::Display;

use schemars::{
    JsonSchema,
    generate::SchemaSettings,
    transform::{Transform, transform_subschemas},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, from_str};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StructuredAnswer {
    #[schemars(
        description = "One or two sentences. If unable_to_answer is true, say what the passages do not cover."
    )]
    pub summary: String,
    #[schemars(
        description = "True when the passages do not contain the answer to the question. Then steps must be empty."
    )]
    pub unable_to_answer: bool,
    #[schemars(
        description = "The procedure in the order it is carried out, one action per step. Empty when unable_to_answer is true."
    )]
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[schemars(description = "One action the operator performs, as a short imperative sentence.")]
    pub instruction: String,
    #[schemars(
        description = "Every WARNING, CAUTION or DANGER from the passages that applies to this step, copied word for word. Empty when there is none."
    )]
    pub warnings: Vec<String>,
    #[schemars(
        description = "The numbers of the passages this step comes from, as numbered in the question message."
    )]
    pub source_indices: Vec<i64>,
}

/// How many characters of a malformed answer an error quotes.
const EXCERPT_CHARS: usize = 120;

#[derive(Error, Debug)]
pub struct AnswerError {
    pub reason: String,
    pub excerpt: String,
}

impl Display for AnswerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the model's answer does not match the schema ({}); it starts with: {}",
            self.reason, self.excerpt
        )
    }
}

// schemars writes "format": "int64" next to every i64, and strict mode does not accept it.
#[derive(Clone)]
struct SchemaFormat;

impl Transform for SchemaFormat {
    fn transform(&mut self, schema: &mut schemars::Schema) {
        schema.remove("format");
        transform_subschemas(self, schema);
    }
}

pub const SCHEMA_NAME: &str = "procedure_answer";

pub fn parse_answer(raw: &str) -> Result<StructuredAnswer, AnswerError> {
    from_str(raw).map_err(|e| AnswerError {
        reason: e.to_string(),
        excerpt: raw.chars().take(EXCERPT_CHARS).collect::<String>(),
    })
}

pub fn answer_schema() -> Value {
    SchemaSettings::draft2020_12()
        .with(|settings| {
            settings.inline_subschemas = true;
            settings.meta_schema = None;
        })
        .with_transform(SchemaFormat)
        .into_generator()
        .into_root_schema_for::<StructuredAnswer>()
        .to_value()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    // ---- parse_answer ----

    #[test]
    fn a_full_answer_parses() {
        let raw = r#"{
            "summary": "Drain the boiler in two steps.",
            "unable_to_answer": false,
            "steps": [
                {
                    "instruction": "Switch off the burner.",
                    "warnings": [],
                    "source_indices": [1]
                },
                {
                    "instruction": "Open the drain valve.",
                    "warnings": ["WARNING: Scalding water may discharge.", "CAUTION: Wear gloves."],
                    "source_indices": [1, 2]
                }
            ]
        }"#;

        let expected = StructuredAnswer {
            summary: "Drain the boiler in two steps.".to_string(),
            unable_to_answer: false,
            steps: vec![
                Step {
                    instruction: "Switch off the burner.".to_string(),
                    warnings: vec![],
                    source_indices: vec![1],
                },
                Step {
                    instruction: "Open the drain valve.".to_string(),
                    warnings: vec![
                        "WARNING: Scalding water may discharge.".to_string(),
                        "CAUTION: Wear gloves.".to_string(),
                    ],
                    source_indices: vec![1, 2],
                },
            ],
        };
        assert_eq!(parse_answer(raw).unwrap(), expected);
    }

    #[test]
    fn an_answer_that_says_it_cannot_answer_parses_with_no_steps() {
        let raw = r#"{"summary": "The passages do not cover the pump.", "unable_to_answer": true, "steps": []}"#;

        let answer = parse_answer(raw).unwrap();

        assert!(answer.unable_to_answer);
        assert!(answer.steps.is_empty());
        assert_eq!(answer.summary, "The passages do not cover the pump.");
    }

    #[test]
    fn whitespace_around_the_json_is_fine() {
        let raw = "\n  {\"summary\": \"s\", \"unable_to_answer\": true, \"steps\": []}\n\n";

        assert!(parse_answer(raw).is_ok());
    }

    // Manuals are full of units. The degree and micro signs must come through unchanged.
    #[test]
    fn text_with_units_is_kept_as_it_is() {
        let raw = json!({
            "summary": "s",
            "unable_to_answer": false,
            "steps": [{
                "instruction": "Heat to 30\u{b0}C and check the 5 \u{b5}m filter.",
                "warnings": [],
                "source_indices": [1]
            }]
        })
        .to_string();

        let answer = parse_answer(&raw).unwrap();

        assert_eq!(
            answer.steps[0].instruction,
            "Heat to 30\u{b0}C and check the 5 \u{b5}m filter."
        );
    }

    // The model's passage numbers are kept exactly as written, even nonsense ones. Dropping
    // the bad ones is the job of the citation check; failing the whole answer here would
    // throw away a good procedure because of one stray number.
    #[test]
    fn indices_are_kept_exactly_as_the_model_wrote_them() {
        let raw = r#"{"summary": "s", "unable_to_answer": false, "steps": [
            {"instruction": "i", "warnings": [], "source_indices": [0, -3, 99999999999]}
        ]}"#;

        let answer = parse_answer(raw).unwrap();

        assert_eq!(answer.steps[0].source_indices, vec![0, -3, 99_999_999_999]);
    }

    // Whether an answer makes sense (a refusal that still lists steps, a step with no
    // source) is for the later checks to judge. The parser only checks the shape.
    #[test]
    fn the_parser_does_not_judge_whether_an_answer_makes_sense() {
        let raw = r#"{"summary": "s", "unable_to_answer": true, "steps": [
            {"instruction": "i", "warnings": [], "source_indices": []}
        ]}"#;

        let answer = parse_answer(raw).unwrap();

        assert!(answer.unable_to_answer);
        assert_eq!(answer.steps.len(), 1);
        assert!(answer.steps[0].source_indices.is_empty());
    }

    #[rstest]
    #[case::plain_text("Sorry, I cannot help with that.")]
    #[case::empty("")]
    #[case::cut_off_json(r#"{"summary": "s", "unable_to_answer": false, "ste"#)]
    #[case::a_missing_field(r#"{"summary": "s", "unable_to_answer": false}"#)]
    #[case::a_wrong_type(r#"{"summary": "s", "unable_to_answer": "no", "steps": []}"#)]
    #[case::steps_that_are_not_a_list(
        r#"{"summary": "s", "unable_to_answer": false, "steps": "none"}"#
    )]
    #[case::a_field_the_schema_never_had(
        r#"{"summary": "s", "unable_to_answer": false, "steps": [], "confidence": 0.9}"#
    )]
    #[case::a_step_missing_its_warnings(
        r#"{"summary": "s", "unable_to_answer": false, "steps": [{"instruction": "i", "source_indices": [1]}]}"#
    )]
    #[case::an_unknown_field_in_a_step(
        r#"{"summary": "s", "unable_to_answer": false, "steps": [{"instruction": "i", "warnings": [], "source_indices": [], "step_number": 1}]}"#
    )]
    fn a_malformed_answer_is_an_error(#[case] raw: &str) {
        assert!(parse_answer(raw).is_err(), "should have been rejected");
    }

    #[test]
    fn the_error_says_what_was_wrong() {
        let raw = r#"{"summary": "s", "unable_to_answer": false}"#;

        let message = parse_answer(raw).unwrap_err().to_string();

        assert!(message.contains("steps"), "got: {message}");
    }

    #[test]
    fn the_error_quotes_the_start_of_the_answer_and_stays_short() {
        let raw = format!("Sorry, no. {}", "x".repeat(5_000));

        let message = parse_answer(&raw).unwrap_err().to_string();

        assert!(message.contains("Sorry, no."), "got: {message}");
        assert!(
            message.len() < 400,
            "the message quoted too much of the answer: {} bytes",
            message.len()
        );
    }

    // Cutting the quoted start by bytes can land inside a character and panic. The one
    // ASCII letter first puts every even byte position in the middle of an 'e acute'.
    #[test]
    fn cutting_the_excerpt_never_splits_a_multibyte_character() {
        let raw = format!("x{}", "\u{e9}".repeat(1_000));

        let message = parse_answer(&raw).unwrap_err().to_string();

        assert!(message.contains(&"\u{e9}".repeat(20)), "got: {message}");
    }

    // ---- the schema ----
    //
    // The checks below read the schema as plain JSON. Keep the schema inline (no `$ref`
    // or `$defs`): the walker below does not follow references.

    // Keywords strict structured outputs do not accept. A conservative list: newer API
    // versions may accept some of them, but nothing in this schema needs them.
    const REJECTED_KEYWORDS: &[&str] = &[
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "patternProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
    ];

    fn property_names(object: &serde_json::Map<String, Value>) -> BTreeSet<String> {
        object
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn required_names(object: &serde_json::Map<String, Value>) -> BTreeSet<String> {
        object
            .get("required")
            .and_then(Value::as_array)
            .map(|required| {
                required
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    // Every way the schema breaks strict mode, each with the path to the schema node.
    fn strict_mode_problems(schema: &Value) -> Vec<String> {
        let mut problems = Vec::new();
        check_node(schema, "$", &mut problems);
        problems
    }

    fn check_node(node: &Value, path: &str, problems: &mut Vec<String>) {
        let Some(object) = node.as_object() else {
            return;
        };

        for keyword in REJECTED_KEYWORDS {
            if object.contains_key(*keyword) {
                problems.push(format!(
                    "{path}: `{keyword}` is not accepted in strict mode"
                ));
            }
        }

        if object.get("type") == Some(&json!("object")) {
            if object.get("additionalProperties") != Some(&json!(false)) {
                problems.push(format!("{path}: additionalProperties must be false"));
            }
            let properties = property_names(object);
            let required = required_names(object);
            if properties != required {
                problems.push(format!(
                    "{path}: required {required:?} must list exactly the properties {properties:?}"
                ));
            }
        }

        // Go into each property's schema and into array items. Not into the `properties`
        // map itself: its keys are field names, not keywords.
        if let Some(properties) = object.get("properties").and_then(Value::as_object) {
            for (name, child) in properties {
                check_node(child, &format!("{path}.{name}"), problems);
            }
        }
        if let Some(items) = object.get("items") {
            check_node(items, &format!("{path}[]"), problems);
        }
    }

    #[test]
    fn the_schema_obeys_strict_mode() {
        assert_eq!(strict_mode_problems(&answer_schema()), Vec::<String>::new());
    }

    // The test above only means something if the check can fail, so show it failing on
    // schemas that each break one rule.
    #[rstest]
    #[case::an_object_without_the_flag(
        json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}),
        "additionalProperties"
    )]
    #[case::a_property_that_is_not_required(
        json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
               "required": ["a"], "additionalProperties": false}),
        "required"
    )]
    #[case::a_keyword_strict_mode_rejects(
        json!({"type": "object", "properties": {"a": {"type": "array", "minItems": 1, "items": {"type": "string"}}},
               "required": ["a"], "additionalProperties": false}),
        "minItems"
    )]
    #[case::a_problem_inside_array_items(
        json!({"type": "object", "properties": {"list": {"type": "array", "items":
               {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}},
               "required": ["list"], "additionalProperties": false}),
        "$.list[]"
    )]
    fn the_schema_check_can_fail(#[case] schema: Value, #[case] expected_problem: &str) {
        let problems = strict_mode_problems(&schema);

        assert!(
            problems.iter().any(|p| p.contains(expected_problem)),
            "expected a problem mentioning {expected_problem}, got {problems:?}"
        );
    }

    #[test]
    fn a_field_called_like_a_keyword_is_not_a_problem() {
        let schema = json!({
            "type": "object",
            "properties": {"format": {"type": "string"}, "pattern": {"type": "string"}},
            "required": ["format", "pattern"],
            "additionalProperties": false
        });

        assert_eq!(strict_mode_problems(&schema), Vec::<String>::new());
    }

    #[test]
    fn the_schema_root_is_an_object() {
        assert_eq!(answer_schema()["type"], "object");
    }

    // The API takes a-z, A-Z, 0-9, underscores and dashes, at most 64 characters.
    #[test]
    fn the_schema_name_is_acceptable_to_the_api() {
        assert!(!SCHEMA_NAME.is_empty() && SCHEMA_NAME.len() <= 64);
        assert!(
            SCHEMA_NAME
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        );
    }

    // Calls `visit` with the path and the schema of every property, however deeply nested
    // (through `properties` and array `items`).
    fn visit_properties(node: &Value, path: &str, visit: &mut impl FnMut(&str, &Value)) {
        if let Some(properties) = node.get("properties").and_then(Value::as_object) {
            for (name, child) in properties {
                let child_path = format!("{path}.{name}");
                visit(&child_path, child);
                visit_properties(child, &child_path, visit);
            }
        }
        if let Some(items) = node.get("items") {
            visit_properties(items, &format!("{path}[]"), visit);
        }
    }

    fn description_of(property: &Value) -> &str {
        property
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    // The model reads the descriptions, so they are part of the prompt: they are where
    // "copy the warning word for word" is said once more, next to the field it is about.
    #[test]
    fn every_field_has_a_description() {
        let mut undescribed = Vec::new();

        visit_properties(&answer_schema(), "$", &mut |path, property| {
            if description_of(property).trim().is_empty() {
                undescribed.push(path.to_owned());
            }
        });

        assert_eq!(undescribed, Vec::<String>::new());
    }

    // A note to self ("TODO: write this") left in a description would be sent to the model
    // on every call. Whole words only, so "mastodon" does not count.
    fn is_placeholder(description: &str) -> bool {
        const MARKERS: [&str; 4] = ["TODO", "FIXME", "TBD", "XXX"];
        description
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| MARKERS.contains(&word.to_uppercase().as_str()))
    }

    #[test]
    fn no_description_is_a_placeholder() {
        let mut placeholders = Vec::new();

        visit_properties(&answer_schema(), "$", &mut |path, property| {
            let text = description_of(property);
            if is_placeholder(text) {
                placeholders.push(format!("{path}: {text}"));
            }
        });

        assert_eq!(placeholders, Vec::<String>::new());
    }

    #[rstest]
    #[case::todo("TODO: write a useful description", true)]
    #[case::lowercase_todo("todo - later", true)]
    #[case::tbd("The steps, TBD", true)]
    #[case::fixme("FIXME", true)]
    #[case::a_word_that_contains_todo("The mastodon in the manual", false)]
    #[case::an_ordinary_description("One action the operator performs.", false)]
    fn the_placeholder_check_can_fail(#[case] description: &str, #[case] expected: bool) {
        assert_eq!(is_placeholder(description), expected);
    }

    fn has_key_anywhere(value: &Value, key: &str) -> bool {
        match value {
            Value::Object(map) => {
                map.contains_key(key) || map.values().any(|child| has_key_anywhere(child, key))
            }
            Value::Array(items) => items.iter().any(|child| has_key_anywhere(child, key)),
            _ => false,
        }
    }

    // The checks above read the schema as plain JSON and do not follow `$ref`. A schema
    // built from references would pass them while the referenced parts went unchecked, so
    // it has to stay inline. schemars writes `$defs`, `$ref` and `$schema` unless it is
    // told not to.
    #[rstest]
    #[case::references("$ref")]
    #[case::definitions("$defs")]
    #[case::the_meta_schema_line("$schema")]
    fn the_schema_is_self_contained(#[case] key: &str) {
        assert!(
            !has_key_anywhere(&answer_schema(), key),
            "the schema contains `{key}`: check inline_subschemas and meta_schema in answer_schema()"
        );
    }

    fn keys(value: &Value) -> BTreeSet<String> {
        value
            .as_object()
            .unwrap_or_else(|| panic!("expected a JSON object, got {value}"))
            .keys()
            .cloned()
            .collect()
    }

    // The Rust types and the schema are written separately, so they can drift apart. A
    // sample answer, serialized, must have exactly the fields the schema lists.
    #[test]
    fn the_types_and_the_schema_name_the_same_fields() {
        let sample = StructuredAnswer {
            summary: String::new(),
            unable_to_answer: false,
            steps: vec![Step {
                instruction: String::new(),
                warnings: vec![],
                source_indices: vec![],
            }],
        };
        let sample = serde_json::to_value(&sample).unwrap();
        let schema = answer_schema();

        assert_eq!(keys(&sample), keys(&schema["properties"]));
        assert_eq!(
            keys(&sample["steps"][0]),
            keys(&schema["properties"]["steps"]["items"]["properties"])
        );
    }
}
