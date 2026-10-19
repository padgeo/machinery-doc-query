use std::fs;

use printpdf::{
    BuiltinFont, Mm, Op, PdfDocument, PdfFontHandle, PdfPage, PdfSaveOptions, Point, Pt, TextItem,
};

pub fn main() {
    gen_boiler_manual();
    gen_pump_manual();
    gen_no_text();
}

fn gen_boiler_manual() {
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
    let bytes = doc
        .with_pages(vec![page1, page2])
        .save(&PdfSaveOptions::default(), &mut Vec::new());

    let write_res = fs::write("./tests/fixtures/pdfs/boiler_x200_manual.pdf", bytes);
    if write_res.is_err() {
        println!("{:?}", write_res.err());
    }
}

fn gen_pump_manual() {
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
    let bytes = doc
        .with_pages(vec![page1])
        .save(&PdfSaveOptions::default(), &mut Vec::new());
    let write_res = fs::write("./tests/fixtures/pdfs/pump_y50_manual.pdf", bytes);
    if write_res.is_err() {
        println!("{:?}", write_res.err());
    }
}

fn gen_no_text() {
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
    let bytes = doc
        .with_pages(vec![page1])
        .save(&PdfSaveOptions::default(), &mut Vec::new());
    let write_res = fs::write("./tests/fixtures/pdfs/no_text_layer.pdf", bytes);
    if write_res.is_err() {
        println!("{:?}", write_res.err());
    }
}
