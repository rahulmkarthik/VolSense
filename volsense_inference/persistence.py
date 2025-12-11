"""
VolSense Persistence Layer
--------------------------
Daily cache manager for storing forecast and signal data.

Provides a simple JSON-based cache that creates a new file per day,
enabling same-day lookups without re-running expensive inference.

Usage
-----
>>> from volsense_inference.persistence import get_daily_cache
>>> cache = get_daily_cache()
>>> cache.store_entry("AAPL", {"signal": "BUY", "z_score": 1.5})
>>> cache.get_valid_entry("AAPL")
{'signal': 'BUY', 'z_score': 1.5}
"""

import os
import json
from datetime import date
from typing import Optional, Dict, Any
from pathlib import Path

# Directory for cache files
ROOT_DIR = Path(os.environ.get("VOLSENSE_CACHE_ROOT", Path.cwd()))
LOG_DIR = ROOT_DIR / "logs"


class DailyCacheManager:
    """
    Manages daily JSON cache files for VolSense forecast/signal data.
    
    Creates a new cache file per day (vol_cache_YYYY-MM-DD.json) and
    provides get/store methods for ticker-keyed data.
    
    Attributes
    ----------
    today : str
        Today's date in ISO format (YYYY-MM-DD).
    filename : str
        Cache filename for today.
    file_path : Path
        Full path to today's cache file.
    """
    
    def __init__(self):
        """Initialize the cache manager and load today's cache."""
        self.today = date.today().isoformat()
        self.filename = f"vol_cache_{self.today}.json"
        self.file_path = LOG_DIR / self.filename
        
        self._ensure_dir_exists()
        self._ensure_file_exists()
        self._cache: Dict[str, Any] = self._load_cache()
    
    def _ensure_dir_exists(self):
        """Create logs directory if it doesn't exist."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _ensure_file_exists(self):
        """Create a fresh cache file for today if missing."""
        if not self.file_path.exists():
            try:
                with open(self.file_path, "w") as f:
                    json.dump({}, f)
            except Exception as e:
                print(f"[WARN] Could not create cache file: {e}")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load the cache from disk."""
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_cache(self):
        """Persist the cache to disk."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            print(f"[WARN] Failed to write cache: {e}")
    
    def get_valid_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a ticker (today only).
        
        :param ticker: Ticker symbol (case-insensitive).
        :return: Cached data dict or None if not found.
        """
        return self._cache.get(ticker.upper())
    
    def store_entry(self, ticker: str, data: Dict[str, Any]):
        """
        Store data for a ticker in today's cache.
        
        :param ticker: Ticker symbol (will be uppercased).
        :param data: Data dict to store.
        """
        self._cache[ticker.upper()] = data
        self._save_cache()
    
    def store_batch(self, entries: Dict[str, Dict[str, Any]]):
        """
        Store multiple ticker entries at once (more efficient).
        
        :param entries: Dict mapping ticker -> data.
        """
        for ticker, data in entries.items():
            self._cache[ticker.upper()] = data
        self._save_cache()
    
    def get_all_entries(self) -> Dict[str, Any]:
        """
        Return all cached entries for today.
        
        :return: Dict mapping ticker -> data.
        """
        return self._cache.copy()
    
    def clear(self):
        """Clear today's cache (in-memory and on disk)."""
        self._cache = {}
        self._save_cache()
    
    def _today_str(self) -> str:
        """Return today's date string."""
        return self.today
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, ticker: str) -> bool:
        """Check if ticker is in cache."""
        return ticker.upper() in self._cache


# Singleton instance
_daily_cache: Optional[DailyCacheManager] = None


def get_daily_cache() -> DailyCacheManager:
    """
    Get the singleton DailyCacheManager instance.
    
    :return: DailyCacheManager instance for today.
    """
    global _daily_cache
    if _daily_cache is None or _daily_cache.today != date.today().isoformat():
        # Create new instance if day has changed
        _daily_cache = DailyCacheManager()
    return _daily_cache
