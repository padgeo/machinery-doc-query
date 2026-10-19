use std::{io::Error, path::PathBuf};

pub fn extract_pages(path: &PathBuf) -> Result<Vec<String>, PdfExtractionError> {
    let bytes = std::fs::read(path).map_err(PdfExtractionError::IoError)?;
    pdf_extract::extract_text_from_mem_by_pages(&bytes).map_err(PdfExtractionError::PdfError)
}

#[derive(thiserror::Error, Debug)]
pub enum PdfExtractionError {
    #[error("pdf extraction error occurred: {0:?}")]
    PdfError(#[source] pdf_extract::OutputError),
    #[error("io error occurred: {0:?}")]
    IoError(#[source] Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = "tests/fixtures/pdfs";

    #[test]
    fn extracts_known_text_and_page_count_from_boiler_manual() {
        let pages = extract_pages(&PathBuf::from(format!("{FIXTURES}/boiler_x200_manual.pdf")))
            .expect("should extract text from a valid pdf");
        assert_eq!(pages.len(), 2);
        assert!(pages[0].contains("WARNING: Allow unit to cool for 30 minutes before servicing."));
        assert!(pages[1].contains("CAUTION: Do not use abrasive cleaners on the sensor housing."));
    }

    #[test]
    fn extracts_all_pages_from_pump_manual() {
        let pages = extract_pages(&PathBuf::from(format!("{FIXTURES}/pump_y50_manual.pdf")))
            .expect("should extract text from a valid pdf");
        assert_eq!(pages.len(), 1);
        assert!(pages[0].contains("Turn off water source"));
        assert!(pages[0].contains("Undo valve"));
        assert!(pages[0].contains("Install pipe"));
    }

    #[test]
    fn no_text_layer_page_extracts_as_empty_not_an_error() {
        let pages = extract_pages(&PathBuf::from(format!("{FIXTURES}/no_text_layer.pdf")))
            .expect("a page with no text layer should still succeed, just with empty text");
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].trim(), "");
    }

    #[test]
    fn missing_file_returns_io_error() {
        let result = extract_pages(&PathBuf::from(format!("{FIXTURES}/does_not_exist.pdf")));
        assert!(matches!(result, Err(PdfExtractionError::IoError(_))));
    }
}
