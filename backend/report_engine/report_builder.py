from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape


class ReportBuilder:

    def build(self, filename, sections):

        styles = getSampleStyleSheet()

        # 🔥 Custom styles (small improvement)
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            spaceAfter=20
        )

        section_title_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            spaceAfter=10
        )

        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["BodyText"],
            spaceAfter=10
        )

        content = []

        # 🔥 Add Main Report Title
        content.append(
            Paragraph("Medical Decision Report", title_style)
        )

        content.append(Spacer(1, 10))

        for title, text in sections:

            # Ensure safe string conversion
            title = escape(str(title))

            # 🔥 Handle multi-line text (important for doctor notes)
            if text is None:
                text = ""
            else:
                text = escape(str(text)).replace("\n", "<br/>")

            # Section Title
            content.append(
                Paragraph(f"<b>{title}</b>", section_title_style)
            )

            # Section Content
            content.append(
                Paragraph(text, body_style)
            )

            content.append(
                Spacer(1, 15)
            )

        # 🔥 Better page layout
        doc = SimpleDocTemplate(
            filename,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=30
        )

        doc.build(content)

        return filename