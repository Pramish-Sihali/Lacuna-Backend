"""
Document parsing service for PDF and DOCX files with OCR support.
"""
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import io
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class DocumentParser:
    """Handles parsing of PDF and DOCX documents with OCR fallback."""

    def __init__(self):
        """Initialize document parser."""
        # Configure Tesseract path if specified
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    async def parse_document(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Parse document and extract text and metadata.

        Args:
            file_path: Path to the document file
            file_type: Type of file (pdf, docx)

        Returns:
            Dictionary with 'text' and 'metadata' keys

        Raises:
            ValueError: If file type is not supported
        """
        file_type = file_type.lower().replace(".", "")

        if file_type == "pdf":
            return await self._parse_pdf(file_path)
        elif file_type in ["docx", "doc"]:
            return await self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    async def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF document with OCR fallback.

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            doc = fitz.open(file_path)
            text_content = []
            metadata = {
                "pages": doc.page_count,
                "author": doc.metadata.get("author", ""),
                "title": doc.metadata.get("title", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
            }

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()

                # If no text found, try OCR
                if not text.strip():
                    logger.info(f"No text found on page {page_num + 1}, attempting OCR")
                    text = await self._ocr_page(page)

                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")

            doc.close()

            return {
                "text": "\n\n".join(text_content),
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    async def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Parse DOCX document.

        Args:
            file_path: Path to DOCX file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            doc = DocxDocument(file_path)

            # Extract text from paragraphs
            text_content = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    if row_text.strip():
                        text_content.append(row_text)

            # Extract metadata
            core_props = doc.core_properties
            metadata = {
                "author": core_props.author or "",
                "title": core_props.title or "",
                "subject": core_props.subject or "",
                "created": str(core_props.created) if core_props.created else "",
                "modified": str(core_props.modified) if core_props.modified else "",
            }

            return {
                "text": "\n\n".join(text_content),
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            raise

    async def _ocr_page(self, page: fitz.Page) -> str:
        """
        Perform OCR on a PDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text from OCR
        """
        try:
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))

            # Perform OCR
            text = pytesseract.image_to_string(image)

            return text

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    async def extract_metadata_only(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Extract only metadata without parsing full text.

        Args:
            file_path: Path to document
            file_type: Type of file

        Returns:
            Metadata dictionary
        """
        file_type = file_type.lower().replace(".", "")

        try:
            if file_type == "pdf":
                doc = fitz.open(file_path)
                metadata = {
                    "pages": doc.page_count,
                    "author": doc.metadata.get("author", ""),
                    "title": doc.metadata.get("title", ""),
                    "subject": doc.metadata.get("subject", ""),
                }
                doc.close()
                return metadata

            elif file_type in ["docx", "doc"]:
                doc = DocxDocument(file_path)
                core_props = doc.core_properties
                return {
                    "author": core_props.author or "",
                    "title": core_props.title or "",
                    "subject": core_props.subject or "",
                }

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
