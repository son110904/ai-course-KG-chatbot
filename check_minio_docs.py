# check_minio_docs.py
"""
Check what documents are available in MinIO
"""

from minio import Minio
from dotenv import load_dotenv
import os
import json

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "203.113.132.48:8008")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "course2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "course2-s3-uiauia")
MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "syllabus")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

MINIO_FOLDERS = [
    "courses-processed/curriculum/",
    "courses-processed/syllabus/",
    "courses-processed/career description/"
]

def check_minio_documents():
    """Check what documents exist in MinIO."""
    
    print("\n" + "="*80)
    print("MINIO DOCUMENTS CHECK")
    print("="*80)
    
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        print(f"\n✅ Connected to MinIO: {MINIO_ENDPOINT}")
        print(f"Bucket: {MINIO_BUCKET}")
        
        all_files = []
        
        for folder in MINIO_FOLDERS:
            print(f"\n📁 Folder: {folder}")
            print("-" * 80)
            
            try:
                objects = client.list_objects(
                    MINIO_BUCKET,
                    prefix=folder,
                    recursive=True
                )
                
                folder_files = []
                for obj in objects:
                    if obj.object_name.lower().endswith(".json"):
                        folder_files.append(obj.object_name)
                
                if folder_files:
                    print(f"Found {len(folder_files)} JSON files:")
                    for file in sorted(folder_files):
                        filename = file.split("/")[-1]
                        print(f"  - {filename}")
                        all_files.append(file)
                else:
                    print("  ❌ No JSON files found in this folder")
                    
            except Exception as e:
                print(f"  ❌ Error accessing folder: {e}")
        
        # Summary
        print(f"\n" + "="*80)
        print(f"SUMMARY")
        print("="*80)
        print(f"Total JSON files found: {len(all_files)}")
        
        # Check for specific files
        print(f"\n🔍 Looking for specific subjects:")
        subjects_to_check = [
            "vật lý",
            "java",
            "lập trình",
            "toán",
            "hóa"
        ]
        
        for subject in subjects_to_check:
            matching = [f for f in all_files if subject.lower() in f.lower()]
            if matching:
                print(f"  ✅ '{subject}': {len(matching)} file(s)")
                for m in matching:
                    print(f"     - {m.split('/')[-1]}")
            else:
                print(f"  ❌ '{subject}': Not found")
        
        # Sample a file to check content
        if all_files:
            print(f"\n📄 SAMPLE FILE CONTENT:")
            print("-" * 80)
            sample_file = all_files[0]
            print(f"File: {sample_file}")
            
            try:
                response = client.get_object(MINIO_BUCKET, sample_file)
                content = response.read()
                data = json.loads(content.decode('utf-8'))
                
                print(f"\nStructure:")
                print(f"  - source_file: {data.get('source_file', 'N/A')}")
                print(f"  - document_type: {data.get('document_type', 'N/A')}")
                
                if 'content' in data:
                    content_data = data['content']
                    print(f"  - paragraphs: {len(content_data.get('paragraphs', []))}")
                    print(f"  - tables: {len(content_data.get('tables', []))}")
                    
                    # Show first few paragraphs
                    paragraphs = content_data.get('paragraphs', [])
                    if paragraphs:
                        print(f"\nFirst 3 paragraphs:")
                        for i, p in enumerate(paragraphs[:3], 1):
                            preview = p[:80] + "..." if len(p) > 80 else p
                            print(f"  {i}. {preview}")
                
                response.close()
                response.release_conn()
                
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
        
    except Exception as e:
        print(f"\n❌ Error connecting to MinIO: {e}")
        print(f"\nPlease check:")
        print(f"  - MINIO_ENDPOINT: {MINIO_ENDPOINT}")
        print(f"  - MINIO_ACCESS_KEY: {MINIO_ACCESS_KEY}")
        print(f"  - MINIO_BUCKET_NAME: {MINIO_BUCKET}")

if __name__ == "__main__":
    check_minio_documents()