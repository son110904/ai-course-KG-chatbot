from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def read_docx_with_tables(file_path):
    """
    Read a .docx file and extract all content including tables.
    
    Args:
        file_path: Path to the .docx file
        
    Returns:
        str: Extracted text with tables formatted as structured text
    """
    doc = Document(file_path)
    full_text = []
    
    for element in doc.element.body:
        # Check if element is a paragraph
        if element.tag.endswith('p'):
            paragraph = Paragraph(element, doc)
            text = paragraph.text.strip()
            if text:
                full_text.append(text)
        
        # Check if element is a table
        elif element.tag.endswith('tbl'):
            table = Table(element, doc)
            table_text = format_table_as_text(table)
            if table_text:
                full_text.append(table_text)
    
    return "\n\n".join(full_text)


def format_table_as_text(table):
    """
    Convert a table into structured text format.
    
    Format:
    [TABLE]
    Header1 | Header2 | Header3
    ---
    Row1Col1 | Row1Col2 | Row1Col3
    Row2Col1 | Row2Col2 | Row2Col3
    [/TABLE]
    
    Args:
        table: python-docx Table object
        
    Returns:
        str: Formatted table as text
    """
    if not table.rows:
        return ""
    
    lines = ["[TABLE]"]
    
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip() for cell in row.cells]
        row_text = " | ".join(cells)
        lines.append(row_text)
        
        # Add separator after first row (assumed to be header)
        if i == 0:
            lines.append("---")
    
    lines.append("[/TABLE]")
    
    return "\n".join(lines)


def read_multiple_docx_files(file_paths):
    """
    Read multiple .docx files and return their contents as a list.
    
    Args:
        file_paths: List of file paths to .docx files
        
    Returns:
        list: List of document contents
    """
    documents = []
    for file_path in file_paths:
        try:
            content = read_docx_with_tables(file_path)
            if content:
                documents.append(content)
                print(f"Successfully read: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return documents


def read_docx_from_directory(directory_path):
    """
    Read all .docx files from a directory.
    
    Args:
        directory_path: Path to directory containing .docx files
        
    Returns:
        list: List of document contents
    """
    import os
    
    docx_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx") and not filename.startswith("~"):
            file_path = os.path.join(directory_path, filename)
            docx_files.append(file_path)
    
    return read_multiple_docx_files(docx_files)