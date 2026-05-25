"""
Document Preprocessor Module - Handles text splitting and cleaning
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class DocumentPreprocessor:
    def __init__(self, chunk_size=200, chunk_overlap=50):
        """
        Initialize the document preprocessor
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        print(f"✅ Document preprocessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        try:
            documents = self.text_splitter.create_documents([text])
            chunks = [doc.page_content for doc in documents]
            return chunks
        except Exception as e:
            print(f"❌ Error splitting text: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        cleaned = " ".join(text.split())
        
        # Remove multiple consecutive newlines
        lines = cleaned.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def process_document(self, text: str) -> List[str]:
        """
        Complete document processing: clean and split
        
        Args:
            text: Raw document text
            
        Returns:
            List of cleaned text chunks
        """
        # First clean the text
        cleaned_text = self.clean_text(text)
        
        # Then split into chunks
        chunks = self.split_text(cleaned_text)
        
        return chunks
    
    def process_from_file(self, file_path: str) -> List[str]:
        """
        Load text from file and process it
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of cleaned text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return self.process_document(text)
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return []
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return []
