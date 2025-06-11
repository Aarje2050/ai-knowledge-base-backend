#!/usr/bin/env python3
"""
Qdrant Inspector - Check document embeddings and manage collections
Run this script to inspect your Qdrant database and see what documents are stored.

Usage:
    python qdrant_inspector.py [command]
    
Commands:
    status     - Show overall Qdrant status
    list       - List all documents and their embeddings
    search     - Search for specific document
    delete     - Delete specific document or all documents
    stats      - Show detailed statistics
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

class QdrantInspector:
    def __init__(self):
        # Initialize Qdrant client based on your settings
        if hasattr(settings, 'qdrant_url') and settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
            print(f"🔗 Connected to Qdrant at: {settings.qdrant_url}")
        else:
            # Default to local Qdrant
            self.client = QdrantClient(host="localhost", port=6333)
            print("🔗 Connected to local Qdrant (localhost:6333)")
        
        self.collection_name = getattr(settings, 'qdrant_collection_name', 'documents')
        print(f"📚 Using collection: {self.collection_name}")
    
    async def check_status(self):
        """Check overall Qdrant status"""
        try:
            # Check if Qdrant is accessible
            collections = self.client.get_collections()
            print(f"\n✅ Qdrant is accessible!")
            print(f"📊 Total collections: {len(collections.collections)}")
            
            # Check if our collection exists
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists:
                collection_info = self.client.get_collection(self.collection_name)
                print(f"✅ Collection '{self.collection_name}' exists")
                print(f"📈 Vector count: {collection_info.points_count}")
                print(f"📏 Vector dimension: {collection_info.config.params.vectors.size}")
                print(f"🔄 Status: {collection_info.status}")
                return True
            else:
                print(f"❌ Collection '{self.collection_name}' does not exist")
                print("💡 No documents have been uploaded yet, or collection name is different")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to Qdrant: {e}")
            print("💡 Make sure Qdrant is running and accessible")
            return False
    
    async def list_documents(self, limit: int = 100):
        """List all documents in the collection"""
        try:
            # Get collection info first
            if not await self.check_status():
                return
            
            print(f"\n📋 Fetching documents (limit: {limit})...")
            
            # Scroll through all points
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't fetch vectors for listing (too large)
            )
            
            if not points:
                print("📭 No documents found in collection")
                return
            
            print(f"\n📚 Found {len(points)} documents:")
            print("-" * 80)
            
            for i, point in enumerate(points, 1):
                payload = point.payload or {}
                
                # Extract key information
                filename = payload.get('filename', 'Unknown')
                company_id = payload.get('company_id', 'Unknown')
                chunk_index = payload.get('chunk_index', 'Unknown')
                page_number = payload.get('page', 'Unknown')
                upload_date = payload.get('created_at', 'Unknown')
                content_preview = (payload.get('content', '')[:100] + '...') if payload.get('content') else 'No content'
                
                print(f"\n{i}. Document ID: {point.id}")
                print(f"   📄 Filename: {filename}")
                print(f"   🏢 Company ID: {company_id}")
                print(f"   📄 Page: {page_number}")
                print(f"   🧩 Chunk: {chunk_index}")
                print(f"   📅 Uploaded: {upload_date}")
                print(f"   📝 Content: {content_preview}")
            
            if next_page_offset:
                print(f"\n💡 There are more documents. Use --limit {limit * 2} to see more.")
                
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
    
    async def search_documents(self, query: str):
        """Search for documents by filename or content"""
        try:
            if not await self.check_status():
                return
            
            print(f"\n🔍 Searching for: '{query}'")
            
            # Search by metadata filter
            search_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchText(text=query)
                    ),
                    models.FieldCondition(
                        key="content",
                        match=models.MatchText(text=query)
                    )
                ]
            )
            
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=50,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                print("❌ No documents found matching your search")
                return
            
            print(f"✅ Found {len(points)} matching documents:")
            print("-" * 60)
            
            for i, point in enumerate(points, 1):
                payload = point.payload or {}
                filename = payload.get('filename', 'Unknown')
                company_id = payload.get('company_id', 'Unknown')
                content_preview = (payload.get('content', '')[:150] + '...') if payload.get('content') else 'No content'
                
                print(f"\n{i}. ID: {point.id}")
                print(f"   📄 File: {filename}")
                print(f"   🏢 Company: {company_id}")
                print(f"   📝 Content: {content_preview}")
                
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
    
    async def delete_documents(self, document_id: Optional[str] = None, filename: Optional[str] = None, 
                             company_id: Optional[str] = None, confirm: bool = False):
        """Delete specific document(s) or all documents"""
        try:
            if not await self.check_status():
                return
            
            if not confirm:
                print("⚠️  This operation will permanently delete documents!")
                print("   Add --confirm flag to proceed")
                return
            
            if document_id:
                # Delete specific document by ID
                print(f"🗑️  Deleting document ID: {document_id}")
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=[document_id])
                )
                print("✅ Document deleted!")
                
            elif filename:
                # Delete all chunks of a specific file
                print(f"🗑️  Deleting all chunks of file: {filename}")
                delete_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
                
                result = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=delete_filter)
                )
                print(f"✅ Deleted all chunks for file '{filename}'")
                
            elif company_id:
                # Delete all documents for a company
                print(f"🗑️  Deleting all documents for company: {company_id}")
                delete_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="company_id",
                            match=models.MatchValue(value=company_id)
                        )
                    ]
                )
                
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=delete_filter)
                )
                print(f"✅ Deleted all documents for company '{company_id}'")
                
            else:
                # Delete entire collection
                print("🗑️  Deleting entire collection...")
                self.client.delete_collection(self.collection_name)
                print("✅ Collection deleted! All documents removed.")
                
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
    
    async def show_stats(self):
        """Show detailed statistics"""
        try:
            if not await self.check_status():
                return
            
            print("\n📊 Detailed Statistics:")
            print("=" * 50)
            
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            print(f"📈 Total vectors: {collection_info.points_count}")
            print(f"📏 Vector dimension: {collection_info.config.params.vectors.size}")
            print(f"🎯 Distance metric: {collection_info.config.params.vectors.distance}")
            print(f"🔄 Status: {collection_info.status}")
            
            # Get sample of documents to analyze
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            if points:
                # Analyze document distribution
                companies = {}
                files = {}
                upload_dates = []
                
                for point in points:
                    payload = point.payload or {}
                    
                    # Count by company
                    company_id = payload.get('company_id', 'Unknown')
                    companies[company_id] = companies.get(company_id, 0) + 1
                    
                    # Count by file
                    filename = payload.get('filename', 'Unknown')
                    files[filename] = files.get(filename, 0) + 1
                    
                    # Collect upload dates
                    upload_date = payload.get('created_at')
                    if upload_date:
                        upload_dates.append(upload_date)
                
                print(f"\n🏢 Companies ({len(companies)}):")
                for company, count in sorted(companies.items()):
                    print(f"   {company}: {count} chunks")
                
                print(f"\n📄 Files ({len(files)}):")
                for filename, count in sorted(files.items()):
                    print(f"   {filename}: {count} chunks")
                
                if upload_dates:
                    print(f"\n📅 Upload period:")
                    print(f"   Earliest: {min(upload_dates)}")
                    print(f"   Latest: {max(upload_dates)}")
                
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")

def print_help():
    """Print help message"""
    print("""
🔍 Qdrant Inspector - Document Management Tool

Usage:
    python qdrant_inspector.py [command] [options]

Commands:
    status                          - Check Qdrant connection and collection status
    list [--limit N]               - List all documents (default limit: 100)
    search "query"                 - Search documents by filename or content
    delete --id DOCUMENT_ID        - Delete specific document by ID
    delete --file "filename"       - Delete all chunks of a specific file
    delete --company COMPANY_ID    - Delete all documents for a company
    delete --all                   - Delete entire collection
    stats                          - Show detailed statistics
    help                          - Show this help message

Examples:
    python qdrant_inspector.py status
    python qdrant_inspector.py list --limit 50
    python qdrant_inspector.py search "policy"
    python qdrant_inspector.py delete --file "employee_handbook.pdf" --confirm
    python qdrant_inspector.py delete --all --confirm
    
⚠️  Note: All delete operations require --confirm flag for safety
""")

async def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    inspector = QdrantInspector()
    
    try:
        if command == "status":
            await inspector.check_status()
            
        elif command == "list":
            limit = 100
            if "--limit" in sys.argv:
                try:
                    limit = int(sys.argv[sys.argv.index("--limit") + 1])
                except (IndexError, ValueError):
                    print("❌ Invalid limit value")
                    return
            await inspector.list_documents(limit)
            
        elif command == "search":
            if len(sys.argv) < 3:
                print("❌ Please provide a search query")
                return
            query = sys.argv[2]
            await inspector.search_documents(query)
            
        elif command == "delete":
            confirm = "--confirm" in sys.argv
            
            if "--id" in sys.argv:
                try:
                    doc_id = sys.argv[sys.argv.index("--id") + 1]
                    await inspector.delete_documents(document_id=doc_id, confirm=confirm)
                except IndexError:
                    print("❌ Please provide document ID")
            elif "--file" in sys.argv:
                try:
                    filename = sys.argv[sys.argv.index("--file") + 1]
                    await inspector.delete_documents(filename=filename, confirm=confirm)
                except IndexError:
                    print("❌ Please provide filename")
            elif "--company" in sys.argv:
                try:
                    company_id = sys.argv[sys.argv.index("--company") + 1]
                    await inspector.delete_documents(company_id=company_id, confirm=confirm)
                except IndexError:
                    print("❌ Please provide company ID")
            elif "--all" in sys.argv:
                await inspector.delete_documents(confirm=confirm)
            else:
                print("❌ Please specify what to delete (--id, --file, --company, or --all)")
                
        elif command == "stats":
            await inspector.show_stats()
            
        elif command == "help":
            print_help()
            
        else:
            print(f"❌ Unknown command: {command}")
            print_help()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())