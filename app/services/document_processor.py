import os
import uuid
import aiofiles
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import io

from app.config import settings
from app.services.ai_service import AIService
from app.services.vector_store import VectorStore

class DocumentProcessor:
    def __init__(self):
        self.ai_service = AIService()
        self.vector_store = VectorStore()
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return file path"""
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(filename)[1]
        safe_filename = f"{file_id}{file_extension}"
        file_path = os.path.join(settings.upload_dir, safe_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return file_path
    
    def extract_text_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from various file types"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self.extract_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self.extract_from_docx(file_path)
            elif file_extension in ['.txt', '.md']:
                return self.extract_from_text(file_path)
            else:
                return self.extract_from_text(file_path)  # Fallback
                
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return []
    
    def extract_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyPDF2"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n[Page {page_num + 1}]\n{page_text}"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
                
                if text.strip():
                    chunks = self.simple_chunk_text(text, {"source": "pdf", "filename": os.path.basename(file_path)})
                else:
                    print("Warning: No text extracted from PDF")
                
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            
        return chunks
    
    def extract_from_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from DOCX using python-docx"""
        chunks = []
        try:
            doc = Document(file_path)
            text = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            if text.strip():
                chunks = self.simple_chunk_text(text, {"source": "docx", "filename": os.path.basename(file_path)})
            else:
                print("Warning: No text extracted from DOCX")
            
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            
        return chunks
    
    def extract_from_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text files"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if text.strip():
                    chunks = self.simple_chunk_text(text, {"source": "text", "filename": os.path.basename(file_path)})
                else:
                    print("Warning: File is empty")
        except UnicodeDecodeError:
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                    chunks = self.simple_chunk_text(text, {"source": "text", "filename": os.path.basename(file_path)})
            except Exception as e:
                print(f"❌ Error reading text file: {e}")
        except Exception as e:
            print(f"❌ Error reading text file: {e}")
            
        return chunks
    
    def simple_chunk_text(self, text: str, base_metadata: Dict = None) -> List[Dict[str, Any]]:
        """Simple text chunking with overlap"""
        if base_metadata is None:
            base_metadata = {}
            
        chunks = []
        
        # Clean text
        text = text.replace('\n\n', '\n').replace('\r', '')
        sentences = text.split('. ')
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '. '.join(current_chunk)
                if not chunk_text.endswith('.'):
                    chunk_text += '.'
                    
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **base_metadata, 
                        "chunk_method": "sentence_based",
                        "chunk_index": len(chunks)
                    }
                })
                
                # Keep some overlap (last few sentences)
                overlap_size = min(3, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:] + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
                
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **base_metadata,
                    "chunk_method": "sentence_based",
                    "chunk_index": len(chunks)
                }
            })
        
        return chunks
    
    async def process_document(self, 
                             file_path: str, 
                             filename: str,
                             company_id: str = "default") -> Dict[str, Any]:
        """Process document: extract text, create embeddings, store in vector DB"""
        try:
            document_id = str(uuid.uuid4())
            
            # Extract text chunks
            print(f"📄 Extracting text from: {filename}")
            text_chunks = self.extract_text_from_file(file_path)
            
            if not text_chunks:
                return {"success": False, "error": "No text could be extracted from the document"}
            
            print(f"📝 Extracted {len(text_chunks)} chunks")
            
            # Generate embeddings for all chunks
            print("🔢 Generating embeddings...")
            chunk_texts = [chunk["content"] for chunk in text_chunks]
            embeddings = await self.ai_service.get_embeddings_batch(chunk_texts)
            
            # Prepare chunks for vector store
            vector_chunks = []
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk_data = {
                    "document_id": document_id,
                    "content": chunk["content"],
                    "embedding": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "filename": filename,
                        "company_id": company_id,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    },
                    "chunk_index": i,
                    "source_type": "custom"
                }
                vector_chunks.append(chunk_data)
            
            # Store in vector database
            print("💾 Storing in vector database...")
            success = await self.vector_store.add_documents(vector_chunks)
            
            if success:
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "chunks_processed": len(text_chunks),
                    "filename": filename
                }
            else:
                return {"success": False, "error": "Failed to store in vector database"}
                
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            return {"success": False, "error": str(e)}