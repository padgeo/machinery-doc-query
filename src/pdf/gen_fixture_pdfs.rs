use std::path::PathBuf;

use printpdf::{
    BuiltinFont, Mm, Op, PdfDocument, PdfFontHandle, PdfPage, PdfSaveOptions, Point, Pt, TextItem,
};
use tempfile::TempDir;

pub const FIXTURES_DIR: &str = "tests/fixtures/pdfs";
pub const BOILER: &str = "boiler_x200_manual.pdf";
pub const PUMP: &str = "pump_y50_manual.pdf";
pub const NO_TEXT: &str = "no_text_layer.pdf";

pub struct TestPdfs {
    dir: TempDir,
}

impl TestPdfs {
    pub fn generate() -> Self {
        let dir = tempfile::tempdir().expect("create a temporary directory");
        std::fs::write(dir.path().join(BOILER), Self::gen_boiler_manual()).unwrap();
        std::fs::write(dir.path().join(PUMP), Self::gen_pump_manual()).unwrap();
        std::fs::write(dir.path().join(NO_TEXT), Self::gen_no_text()).unwrap();
        TestPdfs { dir }
    }

    pub fn dir(&self) -> PathBuf {
        self.dir.path().to_path_buf()
    }
    pub fn path(&self, file_name: &str) -> PathBuf {
        self.dir.path().join(file_name)
    }

    fn gen_boiler_manual() -> Vec<u8> {
        let mut doc = PdfDocument::new("boiler_x200_manual");
        let page1 = PdfPage::new(
            Mm(210.0),
            Mm(297.0),
            vec![
                Op::SaveGraphicsState,
                Op::StartTextSection,
                Op::SetTextCursor {
                    pos: Point::new(Mm(20.0), Mm(270.0)),
                },
                Op::SetFont {
                    font: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                    size: Pt(18.0),
                },
                Op::SetLineHeight { lh: Pt(24.0) },
                Op::ShowText {
                    items: vec![TextItem::Text(
                        "WARNING: Allow unit to cool for 30 minutes before servicing.".to_string(),
                    )],
                },
            ],
        );
        let page2 = PdfPage::new(
            Mm(210.0),
            Mm(297.0),
            vec![
                Op::SaveGraphicsState,
                Op::StartTextSection,
                Op::SetTextCursor {
                    pos: Point::new(Mm(20.0), Mm(270.0)),
                },
                Op::SetFont {
                    font: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                    size: Pt(18.0),
                },
                Op::SetLineHeight { lh: Pt(24.0) },
                Op::ShowText {
                    items: vec![TextItem::Text(
                        "CAUTION: Do not use abrasive cleaners on the sensor housing.".to_string(),
                    )],
                },
                Op::EndTextSection,
                Op::RestoreGraphicsState,
            ],
        );
        doc.with_pages(vec![page1, page2])
            .save(&PdfSaveOptions::default(), &mut Vec::new())
    }

    fn gen_pump_manual() -> Vec<u8> {
        let mut doc = PdfDocument::new("pump_y50_manual");
        let page1 = PdfPage::new(
            Mm(210.0),
            Mm(297.0),
            vec![
                Op::SaveGraphicsState,
                Op::StartTextSection,
                Op::SetTextCursor {
                    pos: Point::new(Mm(20.0), Mm(270.0)),
                },
                Op::SetFont {
                    font: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                    size: Pt(18.0),
                },
                Op::SetLineHeight { lh: Pt(24.0) },
                Op::ShowText {
                    items: vec![TextItem::Text("1. Turn off water source".to_string())],
                },
                Op::AddLineBreak,
                Op::ShowText {
                    items: vec![TextItem::Text("2. Undo valve".to_string())],
                },
                Op::AddLineBreak,
                Op::ShowText {
                    items: vec![TextItem::Text("3. Install pipe".to_string())],
                },
                Op::EndTextSection,
                Op::RestoreGraphicsState,
            ],
        );
        doc.with_pages(vec![page1])
            .save(&PdfSaveOptions::default(), &mut Vec::new())
    }

    fn gen_no_text() -> Vec<u8> {
        let mut doc = PdfDocument::new("no_text_layer");
        let page1 = PdfPage::new(
            Mm(210.0),
            Mm(297.0),
            vec![
                Op::SaveGraphicsState,
                Op::StartTextSection,
                Op::AddLineBreak,
                Op::EndTextSection,
                Op::RestoreGraphicsState,
            ],
        );
        doc.with_pages(vec![page1])
            .save(&PdfSaveOptions::default(), &mut Vec::new())
    }
}
