"""
Enhanced Database Manager with AWS RDS PostgreSQL Cluster Support
"""
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool
import boto3
import json
import time
from dataclasses import dataclass

log = logging.getLogger("database-manager")

@dataclass
class DatabaseConfig:
    """Database configuration from environment"""
    primary_url: str
    reader_url: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    max_connections: int
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            primary_url=os.getenv("DATABASE_URL"),
            reader_url=os.getenv("DATABASE_READER_URL"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "500")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "1000")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            max_connections=int(os.getenv("AWS_RDS_MAX_CONNECTIONS", "1000"))
        )

@dataclass
class AgentExecutionRecord:
    """Agent execution record for tracking"""
    execution_id: str
    agent_id: str
    agent_type: str
    status: str = "pending"
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass 
class SwarmCoordinationRecord:
    """Swarm coordination record for tracking"""
    coordination_id: str
    swarm_id: str
    agent_count: int
    status: str = "pending"
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """Production-grade database manager with AWS RDS integration"""
    
    def __init__(self):
        self.config = DatabaseConfig.from_env()
        self.primary_pool: Optional[Pool] = None
        self.reader_pool: Optional[Pool] = None
        self._initialized = False
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "read_queries": 0,
            "write_queries": 0,
            "connection_errors": 0,
            "query_errors": 0,
            "avg_query_time": 0.0
        }
    
    async def initialize(self):
        """Initialize database pools with AWS RDS cluster"""
        if self._initialized:
            return
            
        try:
            # Create connection pools
            self.primary_pool = await asyncpg.create_pool(
                self.config.primary_url,
                min_size=self.config.pool_size // 4,
                max_size=self.config.pool_size,
                max_queries=50000,
                max_inactive_connection_lifetime=self.config.pool_recycle,
                command_timeout=self.config.pool_timeout,
            )
            
            if self.config.reader_url:
                self.reader_pool = await asyncpg.create_pool(
                    self.config.reader_url,
                    min_size=self.config.pool_size // 4,
                    max_size=self.config.pool_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=self.config.pool_recycle,
                    command_timeout=self.config.pool_timeout,
                )
            else:
                self.reader_pool = self.primary_pool
            
            await self._initialize_schema()
            self._initialized = True
            log.info("Database manager initialized with AWS RDS cluster")
            
        except Exception as e:
            log.error(f"Failed to initialize database manager: {e}")
            raise
    
    async def _initialize_schema(self):
        """Initialize database schema"""
        schema_sql = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
        
        CREATE TABLE IF NOT EXISTS agent_executions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            agent_id VARCHAR(255) NOT NULL,
            swarm_id VARCHAR(255) NOT NULL,
            task_id VARCHAR(255) NOT NULL,
            status VARCHAR(50) NOT NULL,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            result JSONB,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS neural_mesh_memory (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            tier VARCHAR(10) NOT NULL,
            key VARCHAR(500) NOT NULL,
            value TEXT NOT NULL,
            context JSONB,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
        
        CREATE INDEX IF NOT EXISTS idx_agent_executions_swarm_id ON agent_executions(swarm_id);
        CREATE INDEX IF NOT EXISTS idx_neural_mesh_tier_key ON neural_mesh_memory(tier, key);
        """
        
        async with self.primary_pool.acquire() as conn:
            await conn.execute(schema_sql)
            log.info("Database schema initialized")
    
    @asynccontextmanager
    async def get_connection(self, read_only: bool = False):
        """Get database connection with automatic routing"""
        pool = self.reader_pool if read_only else self.primary_pool
        
        async with pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                self.stats["connection_errors"] += 1
                log.error(f"Database connection error: {e}")
                raise
    
    async def execute_query(self, query: str, params: List[Any] = None, read_only: bool = False) -> Any:
        """Execute database query with performance tracking"""
        start_time = time.time()
        
        try:
            async with self.get_connection(read_only=read_only) as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
                
                # Update statistics
                query_time = time.time() - start_time
                self.stats["total_queries"] += 1
                if read_only:
                    self.stats["read_queries"] += 1
                else:
                    self.stats["write_queries"] += 1
                
                return result
                
        except Exception as e:
            self.stats["query_errors"] += 1
            log.error(f"Query execution error: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database manager statistics"""
        return self.stats
    
    def record_request_processing(self, request_id: str, user_id: str, endpoint: str, 
                                 processing_time: float, agents_deployed: int, 
                                 success: bool, capabilities_used: List[str] = None):
        """Record request processing metrics"""
        try:
            # Update statistics
            self.stats["total_requests"] = self.stats.get("total_requests", 0) + 1
            self.stats["total_agents_deployed"] = self.stats.get("total_agents_deployed", 0) + agents_deployed
            self.stats["average_processing_time"] = (
                (self.stats.get("average_processing_time", 0.0) * (self.stats["total_requests"] - 1) + processing_time) 
                / self.stats["total_requests"]
            )
            
            if success:
                self.stats["successful_requests"] = self.stats.get("successful_requests", 0) + 1
            else:
                self.stats["failed_requests"] = self.stats.get("failed_requests", 0) + 1
            
            log.debug(f"Recorded processing for request {request_id}: {processing_time:.3f}s, {agents_deployed} agents")
            
        except Exception as e:
            log.error(f"Failed to record request processing: {e}")
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.primary_pool:
            await self.primary_pool.close()
        
        if self.reader_pool and self.reader_pool != self.primary_pool:
            await self.reader_pool.close()

# Global database manager instance
database_manager = DatabaseManager()

async def get_database_manager() -> DatabaseManager:
    """Get initialized database manager"""
    if not database_manager._initialized:
        await database_manager.initialize()
    return database_manager

def get_db_manager() -> DatabaseManager:
    """Get database manager instance (sync version)"""
    return database_manager