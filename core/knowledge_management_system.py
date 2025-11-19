#!/usr/bin/env python3
"""
Knowledge Management System for AgentForge
Vector database integration, embedding pipeline, semantic search, and RAG implementation
"""

import asyncio
import json
import time
import logging
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle

log = logging.getLogger("knowledge-management-system")

# Vector database integrations
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

class VectorDBProvider(Enum):
    """Supported vector database providers"""
    PINECONE = "pinecone"
    CHROMADB = "chromadb"
    LOCAL = "local"

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    LOCAL = "local"

class DocumentType(Enum):
    """Types of documents that can be processed"""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    CSV = "csv"

@dataclass
class Document:
    """Document for knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: DocumentType = DocumentType.TEXT
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    source: str = ""

@dataclass
class SearchResult:
    """Search result from knowledge base"""
    document: Document
    score: float
    relevance_explanation: str = ""

@dataclass
class RAGResponse:
    """Response from RAG system"""
    query: str
    response: str
    source_documents: List[SearchResult]
    confidence: float
    reasoning: str
    token_usage: int
    processing_time: float

class KnowledgeManagementSystem:
    """Comprehensive knowledge management with vector search and RAG"""
    
    def __init__(
        self,
        vector_db_provider: VectorDBProvider = VectorDBProvider.LOCAL,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    ):
        self.vector_db_provider = vector_db_provider
        self.embedding_provider = embedding_provider
        
        # Storage
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.knowledge_versions: Dict[str, List[str]] = {}
        
        # Vector database clients
        self.pinecone_client = None
        self.chroma_client = None
        
        # Embedding models
        self.sentence_transformer = None
        self.openai_client = None
        
        # Neural mesh integration
        self.neural_mesh = None
        
        # Initialize
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components"""
        try:
            await self._initialize_vector_db()
            await self._initialize_embedding_provider()
            await self._initialize_neural_mesh()
            
            log.info("✅ Knowledge management system initialized")
            
        except Exception as e:
            log.error(f"Failed to initialize knowledge management system: {e}")
    
    async def _initialize_vector_db(self):
        """Initialize vector database"""
        
        if self.vector_db_provider == VectorDBProvider.PINECONE and PINECONE_AVAILABLE:
            try:
                api_key = os.getenv("PINECONE_API_KEY")
                if api_key:
                    pinecone.init(api_key=api_key)
                    
                    # Create or connect to index
                    index_name = "agentforge-knowledge"
                    if index_name not in pinecone.list_indexes():
                        pinecone.create_index(
                            name=index_name,
                            dimension=384,  # sentence-transformers dimension
                            metric="cosine"
                        )
                    
                    self.pinecone_client = pinecone.Index(index_name)
                    log.info("✅ Pinecone vector database initialized")
                else:
                    log.warning("PINECONE_API_KEY not found, falling back to local storage")
                    self.vector_db_provider = VectorDBProvider.LOCAL
                    
            except Exception as e:
                log.error(f"Failed to initialize Pinecone: {e}")
                self.vector_db_provider = VectorDBProvider.LOCAL
        
        elif self.vector_db_provider == VectorDBProvider.CHROMADB and CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.Client()
                self.chroma_collection = self.chroma_client.create_collection(
                    name="agentforge_knowledge",
                    get_or_create=True
                )
                log.info("✅ ChromaDB vector database initialized")
                
            except Exception as e:
                log.error(f"Failed to initialize ChromaDB: {e}")
                self.vector_db_provider = VectorDBProvider.LOCAL
        
        if self.vector_db_provider == VectorDBProvider.LOCAL:
            # Use local storage
            os.makedirs("var/knowledge", exist_ok=True)
            log.info("✅ Local vector storage initialized")
    
    async def _initialize_embedding_provider(self):
        """Initialize embedding provider"""
        
        if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                log.info("✅ Sentence Transformers embedding provider initialized")
            except Exception as e:
                log.error(f"Failed to initialize Sentence Transformers: {e}")
                self.embedding_provider = EmbeddingProvider.LOCAL
        
        elif self.embedding_provider == EmbeddingProvider.OPENAI and OPENAI_EMBEDDINGS_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                    log.info("✅ OpenAI embedding provider initialized")
                else:
                    log.warning("OPENAI_API_KEY not found")
                    self.embedding_provider = EmbeddingProvider.LOCAL
            except Exception as e:
                log.error(f"Failed to initialize OpenAI embeddings: {e}")
                self.embedding_provider = EmbeddingProvider.LOCAL
        
        if self.embedding_provider == EmbeddingProvider.LOCAL:
            log.info("✅ Local embedding provider initialized (hash-based)")
    
    async def _initialize_neural_mesh(self):
        """Initialize neural mesh integration"""
        try:
            from services.neural_mesh.core.enhanced_memory import EnhancedNeuralMesh
            self.neural_mesh = EnhancedNeuralMesh()
            await self.neural_mesh.initialize()
            log.info("✅ Neural mesh integration initialized for knowledge management")
        except ImportError:
            log.warning("Neural mesh not available for knowledge management")
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        document_type: DocumentType = DocumentType.TEXT,
        document_id: Optional[str] = None,
        tags: List[str] = None
    ) -> str:
        """Add document to knowledge base"""
        
        if document_id is None:
            document_id = self._generate_document_id(content)
        
        # Create document
        document = Document(
            id=document_id,
            content=content,
            metadata=metadata or {},
            document_type=document_type,
            tags=tags or [],
            source=metadata.get("source", "unknown") if metadata else "unknown"
        )
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        document.embedding = embedding
        
        # Store document
        self.documents[document_id] = document
        self.embeddings[document_id] = embedding
        
        # Store in vector database
        await self._store_in_vector_db(document)
        
        # Store in neural mesh
        if self.neural_mesh:
            await self.neural_mesh.store_knowledge(
                agent_id="knowledge_system",
                knowledge_type="document",
                data={
                    "document_id": document_id,
                    "content": content[:500],  # Store preview
                    "metadata": metadata,
                    "tags": tags,
                    "document_type": document_type.value
                },
                memory_tier="L4"
            )
        
        log.info(f"Added document {document_id} to knowledge base")
        return document_id
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Search documents using semantic similarity"""
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search in vector database
            if self.vector_db_provider == VectorDBProvider.PINECONE and self.pinecone_client:
                results = await self._search_pinecone(query_embedding, top_k, filter_metadata)
            elif self.vector_db_provider == VectorDBProvider.CHROMADB and self.chroma_client:
                results = await self._search_chromadb(query_embedding, top_k, filter_metadata)
            else:
                results = await self._search_local(query_embedding, top_k, filter_metadata)
            
            # Filter by minimum score
            results = [r for r in results if r.score >= min_score]
            
            # Add relevance explanations
            for result in results:
                result.relevance_explanation = await self._explain_relevance(query, result.document)
            
            return results
            
        except Exception as e:
            log.error(f"Error searching documents: {e}")
            return []
    
    async def generate_rag_response(
        self,
        query: str,
        agent_id: str,
        max_context_docs: int = 5,
        llm_integration = None
    ) -> RAGResponse:
        """Generate response using Retrieval Augmented Generation"""
        
        start_time = time.time()
        
        try:
            # Search for relevant documents
            search_results = await self.search_documents(query, top_k=max_context_docs)
            
            if not search_results:
                # No relevant documents found
                return RAGResponse(
                    query=query,
                    response="I don't have relevant information in my knowledge base to answer this query.",
                    source_documents=[],
                    confidence=0.1,
                    reasoning="No relevant documents found in knowledge base",
                    token_usage=0,
                    processing_time=time.time() - start_time
                )
            
            # Prepare context from retrieved documents
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"Source {i} (relevance: {result.score:.3f}):")
                context_parts.append(f"Content: {result.document.content}")
                context_parts.append(f"Metadata: {json.dumps(result.document.metadata)}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Create RAG prompt
            rag_prompt = f"""Based on the following information from my knowledge base, please answer the user's question:

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Cites specific information from the provided sources
3. Indicates which sources were most relevant
4. Mentions if any important information might be missing
5. Provides a confidence level for your answer

ANSWER:"""
            
            # Generate response using LLM
            if llm_integration is None:
                from core.enhanced_llm_integration import get_llm_integration
                llm_integration = await get_llm_integration()
            
            from core.enhanced_llm_integration import LLMRequest
            
            request = LLMRequest(
                agent_id=agent_id,
                task_type="rag_response",
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.3
            )
            
            llm_response = await llm_integration.generate_response(request)
            
            # Extract confidence from response
            confidence = self._extract_confidence_from_response(llm_response.content)
            
            # Create RAG response
            rag_response = RAGResponse(
                query=query,
                response=llm_response.content,
                source_documents=search_results,
                confidence=confidence,
                reasoning=f"Retrieved {len(search_results)} relevant documents and generated response using {llm_response.provider}",
                token_usage=llm_response.usage.total_tokens,
                processing_time=time.time() - start_time
            )
            
            # Store RAG interaction in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id=agent_id,
                    knowledge_type="rag_interaction",
                    data={
                        "query": query,
                        "response": llm_response.content,
                        "source_count": len(search_results),
                        "confidence": confidence,
                        "token_usage": llm_response.usage.total_tokens
                    },
                    memory_tier="L2"
                )
            
            return rag_response
            
        except Exception as e:
            log.error(f"Error generating RAG response: {e}")
            return RAGResponse(
                query=query,
                response=f"Error generating response: {str(e)}",
                source_documents=[],
                confidence=0.0,
                reasoning=f"RAG generation failed: {str(e)}",
                token_usage=0,
                processing_time=time.time() - start_time
            )
    
    async def process_document_pipeline(
        self,
        file_path: str,
        document_type: DocumentType = None,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Process document through pipeline and add to knowledge base"""
        
        try:
            # Determine document type if not specified
            if document_type is None:
                document_type = self._detect_document_type(file_path)
            
            # Extract content based on document type
            content = await self._extract_document_content(file_path, document_type)
            
            # Chunk document
            chunks = self._chunk_document(content, chunk_size, overlap)
            
            # Process each chunk
            document_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{Path(file_path).stem}_chunk_{i}"
                
                metadata = {
                    "source_file": file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "document_type": document_type.value
                }
                
                doc_id = await self.add_document(
                    content=chunk,
                    metadata=metadata,
                    document_type=document_type,
                    document_id=chunk_id
                )
                
                document_ids.append(doc_id)
            
            log.info(f"Processed document {file_path} into {len(chunks)} chunks")
            return document_ids
            
        except Exception as e:
            log.error(f"Error processing document {file_path}: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        
        try:
            if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS and self.sentence_transformer:
                embedding = self.sentence_transformer.encode(text).tolist()
                return embedding
            
            elif self.embedding_provider == EmbeddingProvider.OPENAI and self.openai_client:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            
            else:
                # Local hash-based embedding (fallback)
                return self._generate_hash_embedding(text)
                
        except Exception as e:
            log.error(f"Error generating embedding: {e}")
            return self._generate_hash_embedding(text)
    
    def _generate_hash_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """Generate hash-based embedding as fallback"""
        
        # Create multiple hash values for different aspects
        hashes = []
        for i in range(dimension // 32):
            hash_input = f"{text}_{i}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            hashes.append(hash_value)
        
        # Convert to normalized float vector
        embedding = []
        for hash_val in hashes:
            for j in range(32):
                bit = (hash_val >> j) & 1
                embedding.append(float(bit * 2 - 1))  # Convert to -1 or 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding[:dimension]
    
    async def _store_in_vector_db(self, document: Document):
        """Store document in vector database"""
        
        if not document.embedding:
            return
        
        try:
            if self.vector_db_provider == VectorDBProvider.PINECONE and self.pinecone_client:
                self.pinecone_client.upsert([
                    {
                        "id": document.id,
                        "values": document.embedding,
                        "metadata": {
                            "content": document.content[:1000],  # Truncate for metadata
                            "document_type": document.document_type.value,
                            "tags": ",".join(document.tags),
                            "created_at": document.created_at
                        }
                    }
                ])
            
            elif self.vector_db_provider == VectorDBProvider.CHROMADB and self.chroma_collection:
                self.chroma_collection.add(
                    ids=[document.id],
                    embeddings=[document.embedding],
                    documents=[document.content],
                    metadatas=[{
                        "document_type": document.document_type.value,
                        "tags": ",".join(document.tags),
                        "created_at": str(document.created_at)
                    }]
                )
            
            else:
                # Local storage
                with open(f"var/knowledge/{document.id}.pkl", 'wb') as f:
                    pickle.dump({
                        "document": document,
                        "embedding": document.embedding
                    }, f)
                    
        except Exception as e:
            log.error(f"Error storing document in vector DB: {e}")
    
    async def _search_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search using Pinecone"""
        
        try:
            query_response = self.pinecone_client.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_metadata,
                include_metadata=True
            )
            
            results = []
            for match in query_response.matches:
                document_id = match.id
                if document_id in self.documents:
                    result = SearchResult(
                        document=self.documents[document_id],
                        score=match.score
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            log.error(f"Error searching Pinecone: {e}")
            return []
    
    async def _search_local(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_metadata: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search using local vector storage"""
        
        try:
            # Calculate similarities
            similarities = []
            
            for doc_id, doc_embedding in self.embeddings.items():
                if doc_id not in self.documents:
                    continue
                
                document = self.documents[doc_id]
                
                # Apply metadata filter
                if filter_metadata:
                    if not all(
                        document.metadata.get(k) == v 
                        for k, v in filter_metadata.items()
                    ):
                        continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((doc_id, similarity))
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:top_k]
            
            # Create search results
            results = []
            for doc_id, score in top_similarities:
                result = SearchResult(
                    document=self.documents[doc_id],
                    score=score
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            log.error(f"Error in local search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _extract_document_content(
        self,
        file_path: str,
        document_type: DocumentType
    ) -> str:
        """Extract content from document based on type"""
        
        if document_type == DocumentType.TEXT or document_type == DocumentType.MARKDOWN:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif document_type == DocumentType.PDF:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    return content
            except ImportError:
                log.error("PyPDF2 not available for PDF processing")
                return ""
        
        elif document_type == DocumentType.CODE:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Add code-specific metadata
                return f"```{Path(file_path).suffix[1:]}\n{content}\n```"
        
        elif document_type == DocumentType.JSON:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        elif document_type == DocumentType.CSV:
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            except ImportError:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        else:
            # Default to text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension"""
        
        suffix = Path(file_path).suffix.lower()
        
        type_mapping = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.pdf': DocumentType.PDF,
            '.py': DocumentType.CODE,
            '.js': DocumentType.CODE,
            '.ts': DocumentType.CODE,
            '.java': DocumentType.CODE,
            '.cpp': DocumentType.CODE,
            '.c': DocumentType.CODE,
            '.json': DocumentType.JSON,
            '.csv': DocumentType.CSV
        }
        
        return type_mapping.get(suffix, DocumentType.TEXT)
    
    def _chunk_document(
        self,
        content: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Chunk document into smaller pieces with overlap"""
        
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings near the chunk boundary
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                best_break = end
                
                for ending in sentence_endings:
                    pos = content.rfind(ending, start + chunk_size - 200, end)
                    if pos > start:
                        best_break = pos + len(ending)
                        break
                
                chunk = content[start:best_break].strip()
            else:
                chunk = content[start:].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)
            
            if start >= len(content):
                break
        
        return chunks
    
    async def _explain_relevance(self, query: str, document: Document) -> str:
        """Generate explanation for why document is relevant"""
        
        # Simple keyword-based relevance explanation
        query_words = set(query.lower().split())
        doc_words = set(document.content.lower().split())
        
        common_words = query_words.intersection(doc_words)
        
        if common_words:
            return f"Contains keywords: {', '.join(list(common_words)[:5])}"
        else:
            return "Semantic similarity based on content"
    
    def _extract_confidence_from_response(self, response: str) -> float:
        """Extract confidence score from RAG response"""
        
        import re
        
        # Look for confidence indicators
        confidence_patterns = [
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*confidence',
            r'([0-9.]+)%\s*confident',
            r'confidence.*?([0-9.]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    value = float(match.group(1))
                    return min(value if value <= 1.0 else value / 100.0, 1.0)
                except ValueError:
                    continue
        
        # Default confidence based on response quality indicators
        if "uncertain" in response.lower() or "not sure" in response.lower():
            return 0.4
        elif "confident" in response.lower() or "certain" in response.lower():
            return 0.9
        else:
            return 0.7
    
    def _generate_document_id(self, content: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def update_knowledge_version(
        self,
        knowledge_domain: str,
        updates: List[Dict[str, Any]],
        version_type: str = "minor"
    ) -> str:
        """Update knowledge base with versioning"""
        
        try:
            # Generate new version
            current_version = self.knowledge_versions.get(knowledge_domain, ["1.0.0"])[-1]
            new_version = self._increment_knowledge_version(current_version, version_type)
            
            # Apply updates
            updated_documents = []
            for update in updates:
                if update["action"] == "add":
                    doc_id = await self.add_document(
                        content=update["content"],
                        metadata=update.get("metadata", {}),
                        tags=[knowledge_domain, f"version:{new_version}"]
                    )
                    updated_documents.append(doc_id)
                
                elif update["action"] == "update":
                    doc_id = update["document_id"]
                    if doc_id in self.documents:
                        # Create new version of document
                        old_doc = self.documents[doc_id]
                        new_doc_id = f"{doc_id}_v{new_version}"
                        
                        await self.add_document(
                            content=update["content"],
                            metadata={**old_doc.metadata, "previous_version": doc_id},
                            document_type=old_doc.document_type,
                            document_id=new_doc_id,
                            tags=old_doc.tags + [f"version:{new_version}"]
                        )
                        updated_documents.append(new_doc_id)
                
                elif update["action"] == "delete":
                    doc_id = update["document_id"]
                    if doc_id in self.documents:
                        # Mark as deleted rather than actually deleting
                        self.documents[doc_id].metadata["deleted"] = True
                        self.documents[doc_id].metadata["deleted_version"] = new_version
            
            # Record version
            if knowledge_domain not in self.knowledge_versions:
                self.knowledge_versions[knowledge_domain] = []
            self.knowledge_versions[knowledge_domain].append(new_version)
            
            # Store version info in neural mesh
            if self.neural_mesh:
                await self.neural_mesh.store_knowledge(
                    agent_id="knowledge_system",
                    knowledge_type="knowledge_version",
                    data={
                        "domain": knowledge_domain,
                        "version": new_version,
                        "updates_count": len(updates),
                        "updated_documents": updated_documents,
                        "timestamp": time.time()
                    },
                    memory_tier="L4"
                )
            
            log.info(f"Updated knowledge domain {knowledge_domain} to version {new_version}")
            return new_version
            
        except Exception as e:
            log.error(f"Error updating knowledge version: {e}")
            return current_version
    
    def _increment_knowledge_version(self, current_version: str, version_type: str) -> str:
        """Increment knowledge version"""
        
        try:
            parts = current_version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if version_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif version_type == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
            
            return f"{major}.{minor}.{patch}"
            
        except (ValueError, IndexError):
            return "1.0.0"
    
    def get_knowledge_analytics(self) -> Dict[str, Any]:
        """Get knowledge base analytics"""
        
        total_documents = len(self.documents)
        
        # Document type breakdown
        type_breakdown = {}
        for doc in self.documents.values():
            doc_type = doc.document_type.value
            type_breakdown[doc_type] = type_breakdown.get(doc_type, 0) + 1
        
        # Tag analysis
        all_tags = []
        for doc in self.documents.values():
            all_tags.extend(doc.tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Version analysis
        version_info = {
            domain: {
                "current_version": versions[-1],
                "total_versions": len(versions)
            }
            for domain, versions in self.knowledge_versions.items()
        }
        
        return {
            "total_documents": total_documents,
            "document_types": type_breakdown,
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "knowledge_domains": version_info,
            "vector_db_provider": self.vector_db_provider.value,
            "embedding_provider": self.embedding_provider.value,
            "storage_size_mb": self._calculate_storage_size()
        }
    
    def _calculate_storage_size(self) -> float:
        """Calculate approximate storage size in MB"""
        
        total_chars = sum(len(doc.content) for doc in self.documents.values())
        embedding_size = len(self.embeddings) * 384 * 4  # 384 dimensions * 4 bytes per float
        
        return (total_chars + embedding_size) / (1024 * 1024)
    
    async def export_knowledge_base(self, export_path: str = "exports/knowledge"):
        """Export entire knowledge base"""
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export documents
        documents_data = []
        for doc in self.documents.values():
            doc_data = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "document_type": doc.document_type.value,
                "tags": doc.tags,
                "created_at": doc.created_at,
                "version": doc.version
            }
            documents_data.append(doc_data)
        
        with open(export_dir / "documents.json", 'w') as f:
            json.dump(documents_data, f, indent=2)
        
        # Export embeddings
        with open(export_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Export analytics
        analytics = self.get_knowledge_analytics()
        with open(export_dir / "analytics.json", 'w') as f:
            json.dump(analytics, f, indent=2)
        
        log.info(f"Knowledge base exported to {export_path}")

# Global instance
knowledge_system = KnowledgeManagementSystem()
