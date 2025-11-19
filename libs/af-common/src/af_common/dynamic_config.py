"""
Dynamic configuration management for AgentForge services
Supports hot-reloading of configuration without service restarts
"""
from __future__ import annotations

import os
import json
import asyncio
import threading
import weakref
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .config import BaseConfig, get_config
from .logging import get_logger

logger = get_logger("dynamic_config")

@dataclass
class ConfigChange:
    """Represents a configuration change event"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str
    service_name: str

class ConfigWatcher(ABC):
    """Abstract base class for configuration watchers"""
    
    @abstractmethod
    async def start_watching(self) -> None:
        """Start watching for configuration changes"""
        pass
    
    @abstractmethod
    async def stop_watching(self) -> None:
        """Stop watching for configuration changes"""
        pass
    
    @abstractmethod
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        pass

class FileConfigWatcher(ConfigWatcher):
    """Watch configuration files for changes"""
    
    def __init__(self, file_paths: List[str], poll_interval: float = 1.0):
        self.file_paths = [Path(p) for p in file_paths]
        self.poll_interval = poll_interval
        self.last_modified = {}
        self.watching = False
        self._watch_task = None
        
    async def start_watching(self) -> None:
        """Start watching configuration files"""
        self.watching = True
        
        # Initialize last modified times
        for file_path in self.file_paths:
            if file_path.exists():
                self.last_modified[file_path] = file_path.stat().st_mtime
        
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Started watching {len(self.file_paths)} configuration files")
    
    async def stop_watching(self) -> None:
        """Stop watching configuration files"""
        self.watching = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped watching configuration files")
    
    async def _watch_loop(self) -> None:
        """Main watching loop"""
        while self.watching:
            try:
                for file_path in self.file_paths:
                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        last_mtime = self.last_modified.get(file_path, 0)
                        
                        if current_mtime > last_mtime:
                            logger.info(f"Configuration file changed: {file_path}")
                            self.last_modified[file_path] = current_mtime
                            # File changed - trigger reload
                            await self._on_file_changed(file_path)
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in config watch loop: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _on_file_changed(self, file_path: Path) -> None:
        """Handle file change event"""
        # This will be called by the DynamicConfigManager
        pass
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration from files"""
        config = {}
        
        for file_path in self.file_paths:
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        if file_path.suffix.lower() == '.json':
                            file_config = json.load(f)
                        else:
                            # Treat as environment file
                            file_config = {}
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#') and '=' in line:
                                    key, value = line.split('=', 1)
                                    file_config[key.strip()] = value.strip()
                    
                    config.update(file_config)
                    
                except Exception as e:
                    logger.error(f"Failed to load config from {file_path}: {e}")
        
        return config

class EnvironmentConfigWatcher(ConfigWatcher):
    """Watch environment variables for changes (limited functionality)"""
    
    def __init__(self, watch_prefix: str = "AF_"):
        self.watch_prefix = watch_prefix
        self.last_env = {}
        self.watching = False
        self._watch_task = None
    
    async def start_watching(self) -> None:
        """Start watching environment variables"""
        self.watching = True
        self.last_env = {k: v for k, v in os.environ.items() if k.startswith(self.watch_prefix)}
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Started watching environment variables with prefix {self.watch_prefix}")
    
    async def stop_watching(self) -> None:
        """Stop watching environment variables"""
        self.watching = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped watching environment variables")
    
    async def _watch_loop(self) -> None:
        """Watch for environment variable changes"""
        while self.watching:
            try:
                current_env = {k: v for k, v in os.environ.items() if k.startswith(self.watch_prefix)}
                
                # Check for changes
                if current_env != self.last_env:
                    logger.info("Environment variables changed")
                    self.last_env = current_env
                    # Environment changed - trigger reload
                    await self._on_env_changed()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in environment watch loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _on_env_changed(self) -> None:
        """Handle environment change event"""
        # This will be called by the DynamicConfigManager
        pass
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current environment configuration"""
        return {k: v for k, v in os.environ.items() if k.startswith(self.watch_prefix)}

class DynamicConfigManager:
    """Manages dynamic configuration updates for AgentForge services"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, service_name: str = "agentforge"):
        if hasattr(self, '_initialized'):
            return
            
        self.service_name = service_name
        self.watchers: List[ConfigWatcher] = []
        self.subscribers: Set[weakref.ref] = set()
        self.current_config: Dict[str, Any] = {}
        self.config_history: List[ConfigChange] = []
        self.max_history = 100
        self._running = False
        self._lock = asyncio.Lock()
        self._initialized = True
        
        logger.info(f"Initialized DynamicConfigManager for {service_name}")
    
    def add_watcher(self, watcher: ConfigWatcher) -> None:
        """Add a configuration watcher"""
        self.watchers.append(watcher)
        logger.info(f"Added config watcher: {type(watcher).__name__}")
    
    def subscribe(self, callback: Callable[[ConfigChange], None]) -> None:
        """Subscribe to configuration changes"""
        self.subscribers.add(weakref.ref(callback))
        logger.info(f"Added config change subscriber")
    
    async def start(self) -> None:
        """Start dynamic configuration management"""
        if self._running:
            return
        
        self._running = True
        
        # Load initial configuration
        await self.reload_config()
        
        # Start all watchers
        for watcher in self.watchers:
            try:
                await watcher.start_watching()
            except Exception as e:
                logger.error(f"Failed to start watcher {type(watcher).__name__}: {e}")
        
        logger.info("Started dynamic configuration management")
    
    async def stop(self) -> None:
        """Stop dynamic configuration management"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all watchers
        for watcher in self.watchers:
            try:
                await watcher.stop_watching()
            except Exception as e:
                logger.error(f"Failed to stop watcher {type(watcher).__name__}: {e}")
        
        logger.info("Stopped dynamic configuration management")
    
    async def reload_config(self) -> bool:
        """Reload configuration from all sources"""
        async with self._lock:
            try:
                new_config = {}
                
                # Merge configuration from all watchers
                for watcher in self.watchers:
                    try:
                        watcher_config = watcher.get_current_config()
                        new_config.update(watcher_config)
                    except Exception as e:
                        logger.error(f"Failed to get config from watcher {type(watcher).__name__}: {e}")
                
                # Detect changes
                changes = self._detect_changes(self.current_config, new_config)
                
                if changes:
                    logger.info(f"Configuration changes detected: {len(changes)} changes")
                    
                    # Update current config
                    self.current_config = new_config.copy()
                    
                    # Record changes
                    for change in changes:
                        self.config_history.append(change)
                        if len(self.config_history) > self.max_history:
                            self.config_history.pop(0)
                    
                    # Notify subscribers
                    await self._notify_subscribers(changes)
                    
                    return True
                else:
                    logger.debug("No configuration changes detected")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
                return False
    
    def _detect_changes(
        self, 
        old_config: Dict[str, Any], 
        new_config: Dict[str, Any]
    ) -> List[ConfigChange]:
        """Detect changes between old and new configuration"""
        changes = []
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=datetime.now(),
                    source="dynamic_reload",
                    service_name=self.service_name
                )
                changes.append(change)
        
        return changes
    
    async def _notify_subscribers(self, changes: List[ConfigChange]) -> None:
        """Notify all subscribers of configuration changes"""
        # Clean up dead references
        self.subscribers = {ref for ref in self.subscribers if ref() is not None}
        
        for ref in self.subscribers:
            callback = ref()
            if callback:
                try:
                    for change in changes:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(change)
                        else:
                            callback(change)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get current configuration value"""
        return self.current_config.get(key, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all current configuration"""
        return self.current_config.copy()
    
    def get_config_history(self, limit: Optional[int] = None) -> List[ConfigChange]:
        """Get configuration change history"""
        if limit:
            return self.config_history[-limit:]
        return self.config_history.copy()
    
    async def update_config(
        self, 
        updates: Dict[str, Any], 
        source: str = "api"
    ) -> bool:
        """Update configuration programmatically"""
        async with self._lock:
            try:
                changes = []
                
                for key, new_value in updates.items():
                    old_value = self.current_config.get(key)
                    if old_value != new_value:
                        change = ConfigChange(
                            key=key,
                            old_value=old_value,
                            new_value=new_value,
                            timestamp=datetime.now(),
                            source=source,
                            service_name=self.service_name
                        )
                        changes.append(change)
                        self.current_config[key] = new_value
                
                if changes:
                    # Record changes
                    for change in changes:
                        self.config_history.append(change)
                        if len(self.config_history) > self.max_history:
                            self.config_history.pop(0)
                    
                    # Notify subscribers
                    await self._notify_subscribers(changes)
                    
                    logger.info(f"Updated {len(changes)} configuration values")
                    return True
                else:
                    logger.debug("No configuration changes to apply")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                return False

# Global instance
_config_manager: Optional[DynamicConfigManager] = None

def get_dynamic_config_manager(service_name: str = "agentforge") -> DynamicConfigManager:
    """Get or create the global dynamic configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager(service_name)
    return _config_manager

async def setup_dynamic_config(
    service_name: str = "agentforge",
    config_files: Optional[List[str]] = None,
    watch_env: bool = True,
    env_prefix: str = "AF_"
) -> DynamicConfigManager:
    """Setup dynamic configuration management"""
    manager = get_dynamic_config_manager(service_name)
    
    # Add file watcher if config files specified
    if config_files:
        file_watcher = FileConfigWatcher(config_files)
        manager.add_watcher(file_watcher)
    
    # Add environment watcher if enabled
    if watch_env:
        env_watcher = EnvironmentConfigWatcher(env_prefix)
        manager.add_watcher(env_watcher)
    
    # Start the manager
    await manager.start()
    
    return manager

# Context manager for temporary configuration changes
class TemporaryConfigChange:
    """Context manager for temporary configuration changes"""
    
    def __init__(self, manager: DynamicConfigManager, **changes):
        self.manager = manager
        self.changes = changes
        self.original_values = {}
    
    async def __aenter__(self):
        # Store original values
        for key in self.changes.keys():
            self.original_values[key] = self.manager.get_config_value(key)
        
        # Apply changes
        await self.manager.update_config(self.changes, source="temporary")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        await self.manager.update_config(self.original_values, source="restore")
