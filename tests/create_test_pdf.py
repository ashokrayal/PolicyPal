"""
Script to create a simple test PDF for testing the PDF parser.
"""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pathlib import Path

def create_test_pdf():
    """Create a simple test PDF with some text."""
    test_dir = Path("tests/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = test_dir / "test_document.pdf"
    
    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Test Policy Document")
    
    # Add content
    c.setFont("Helvetica", 12)
    content = [
        "This is a test policy document for the PolicyPal project.",
        "",
        "Section 1: Introduction",
        "This document contains sample policy information that can be used",
        "to test the PDF parsing functionality of our RAG chatbot.",
        "",
        "Section 2: Vacation Policy",
        "Employees are entitled to 20 vacation days per year.",
        "Vacation requests must be submitted at least 2 weeks in advance.",
        "",
        "Section 3: Remote Work Policy",
        "Employees may work remotely up to 3 days per week.",
        "Remote work requires manager approval and stable internet connection."
    ]
    
    y_position = height - 150
    for line in content:
        c.drawString(100, y_position, line)
        y_position -= 20
    
    c.save()
    print(f"Test PDF created at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_test_pdf() 