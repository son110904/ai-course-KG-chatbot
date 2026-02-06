import io
from dataclasses import dataclass, field
from minio import Minio
from docx import Document
from pypdf import PdfReader
from config import *

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

@dataclass
class LoadedDocument:
    content: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

def read_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        if p.extract_text():
            pages.append(p.extract_text())
    return "\n".join(pages)

def load_documents():
    documents = []

    for obj in client.list_objects(
        MINIO_BUCKET,
        prefix=MINIO_PREFIX,
        recursive=True
    ):
        if not obj.object_name.endswith((".docx", ".pdf")):
            continue

        data = client.get_object(
            MINIO_BUCKET,
            obj.object_name
        ).read()

        if obj.object_name.endswith(".docx"):
            text = read_docx(data)
        else:
            text = read_pdf(data)

        documents.append(LoadedDocument(
            content=text,
            source=obj.object_name,
            metadata={"bucket": MINIO_BUCKET}
        ))

    return documents
