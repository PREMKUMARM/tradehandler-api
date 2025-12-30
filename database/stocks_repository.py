"""
Repository for selected stocks operations
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from database.stocks_connection import get_stocks_database


class SelectedStock:
    """Model for selected stock"""
    def __init__(
        self,
        tradingsymbol: str,
        exchange: str,
        instrument_token: int,
        instrument_key: str,
        name: Optional[str] = None,
        instrument_type: Optional[str] = None,
        is_active: bool = True,
        notes: Optional[str] = None,
        id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = id
        self.tradingsymbol = tradingsymbol
        self.exchange = exchange
        self.instrument_token = instrument_token
        self.instrument_key = instrument_key
        self.name = name
        self.instrument_type = instrument_type
        self.is_active = is_active
        self.notes = notes
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "tradingsymbol": self.tradingsymbol,
            "exchange": self.exchange,
            "instrument_token": self.instrument_token,
            "instrument_key": self.instrument_key,
            "name": self.name,
            "instrument_type": self.instrument_type,
            "is_active": self.is_active,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_row(cls, row) -> 'SelectedStock':
        """Create from database row"""
        # sqlite3.Row supports dictionary-style access
        # Check if key exists using 'in' operator
        def get_value(key, default=None):
            try:
                # sqlite3.Row supports 'in' operator to check key existence
                if key in row.keys():
                    return row[key]
                return default
            except (KeyError, TypeError, AttributeError):
                return default
        
        return cls(
            id=row["id"],
            tradingsymbol=row["tradingsymbol"],
            exchange=row["exchange"],
            instrument_token=row["instrument_token"],
            instrument_key=row["instrument_key"],
            name=get_value("name"),
            instrument_type=get_value("instrument_type"),
            is_active=bool(row["is_active"]),
            notes=get_value("notes"),
            created_at=datetime.fromisoformat(row["created_at"]) if get_value("created_at") else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if get_value("updated_at") else None
        )


class StocksRepository:
    """Repository for selected stocks operations"""

    def save(self, stock: SelectedStock) -> bool:
        """Save or update a selected stock"""
        try:
            db = get_stocks_database()
            
            # Check if stock already exists by instrument_key
            existing = self.get_by_instrument_key(stock.instrument_key)
            
            # Also check by tradingsymbol+exchange (for UNIQUE constraint)
            if not existing:
                existing = self.get_by_tradingsymbol_exchange(stock.tradingsymbol, stock.exchange)
            
            if existing:
                # Update existing stock
                query = '''
                    UPDATE selected_stocks
                    SET tradingsymbol = ?, exchange = ?, instrument_token = ?,
                        instrument_key = ?, name = ?, instrument_type = ?, is_active = ?,
                        updated_at = ?, notes = ?
                    WHERE id = ?
                '''
                params = (
                    stock.tradingsymbol,
                    stock.exchange,
                    stock.instrument_token,
                    stock.instrument_key,
                    stock.name,
                    stock.instrument_type,
                    1 if stock.is_active else 0,
                    datetime.now().isoformat(),
                    stock.notes,
                    existing.id
                )
            else:
                # Insert new stock
                query = '''
                    INSERT INTO selected_stocks
                    (tradingsymbol, exchange, instrument_token, instrument_key,
                     name, instrument_type, is_active, created_at, updated_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                params = (
                    stock.tradingsymbol,
                    stock.exchange,
                    stock.instrument_token,
                    stock.instrument_key,
                    stock.name,
                    stock.instrument_type,
                    1 if stock.is_active else 0,
                    stock.created_at.isoformat(),
                    stock.updated_at.isoformat(),
                    stock.notes
                )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            error_msg = str(e)
            # Handle UNIQUE constraint violation - stock already exists
            if "UNIQUE constraint" in error_msg or "UNIQUE constraint failed" in error_msg:
                # Try to get and return the existing stock
                existing = self.get_by_instrument_key(stock.instrument_key)
                if not existing:
                    existing = self.get_by_tradingsymbol_exchange(stock.tradingsymbol, stock.exchange)
                if existing:
                    # Update the existing stock
                    return self._update_existing_stock(existing.id, stock)
            print(f"Error saving stock: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _update_existing_stock(self, stock_id: int, stock: SelectedStock) -> bool:
        """Update an existing stock by ID"""
        try:
            db = get_stocks_database()
            query = '''
                UPDATE selected_stocks
                SET tradingsymbol = ?, exchange = ?, instrument_token = ?,
                    instrument_key = ?, name = ?, instrument_type = ?, is_active = ?,
                    updated_at = ?, notes = ?
                WHERE id = ?
            '''
            params = (
                stock.tradingsymbol,
                stock.exchange,
                stock.instrument_token,
                stock.instrument_key,
                stock.name,
                stock.instrument_type,
                1 if stock.is_active else 0,
                datetime.now().isoformat(),
                stock.notes,
                stock_id
            )
            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error updating stock: {e}")
            return False

    def get_all(self, active_only: bool = False) -> List[SelectedStock]:
        """Get all selected stocks"""
        try:
            db = get_stocks_database()
            if active_only:
                query = "SELECT * FROM selected_stocks WHERE is_active = 1 ORDER BY tradingsymbol"
            else:
                query = "SELECT * FROM selected_stocks ORDER BY tradingsymbol"
            
            cursor = db.execute_query(query)
            rows = cursor.fetchall()
            return [SelectedStock.from_row(row) for row in rows]
        except Exception as e:
            print(f"Error getting stocks: {e}")
            return []

    def get_by_instrument_key(self, instrument_key: str) -> Optional[SelectedStock]:
        """Get stock by instrument key"""
        try:
            db = get_stocks_database()
            cursor = db.execute_query(
                "SELECT * FROM selected_stocks WHERE instrument_key = ?",
                (instrument_key,)
            )
            row = cursor.fetchone()
            if row:
                return SelectedStock.from_row(row)
            return None
        except Exception as e:
            print(f"Error getting stock by instrument_key: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_by_tradingsymbol_exchange(self, tradingsymbol: str, exchange: str) -> Optional[SelectedStock]:
        """Get stock by tradingsymbol and exchange"""
        try:
            db = get_stocks_database()
            cursor = db.execute_query(
                "SELECT * FROM selected_stocks WHERE tradingsymbol = ? AND exchange = ?",
                (tradingsymbol, exchange)
            )
            row = cursor.fetchone()
            if row:
                return SelectedStock.from_row(row)
            return None
        except Exception as e:
            print(f"Error getting stock by tradingsymbol/exchange: {e}")
            return None

    def delete(self, instrument_key: str) -> bool:
        """Delete a selected stock"""
        try:
            db = get_stocks_database()
            db.execute_query(
                "DELETE FROM selected_stocks WHERE instrument_key = ?",
                (instrument_key,)
            )
            db.commit()
            return True
        except Exception as e:
            print(f"Error deleting stock: {e}")
            return False

    def toggle_active(self, instrument_key: str, is_active: bool) -> bool:
        """Toggle active status of a stock"""
        try:
            db = get_stocks_database()
            db.execute_query(
                "UPDATE selected_stocks SET is_active = ?, updated_at = ? WHERE instrument_key = ?",
                (1 if is_active else 0, datetime.now().isoformat(), instrument_key)
            )
            db.commit()
            return True
        except Exception as e:
            print(f"Error toggling stock active status: {e}")
            return False

    def bulk_save(self, stocks: List[SelectedStock]) -> int:
        """Bulk save multiple stocks"""
        saved_count = 0
        for stock in stocks:
            if self.save(stock):
                saved_count += 1
        return saved_count


# Global repository instance
_stocks_repo: Optional[StocksRepository] = None


def get_stocks_repository() -> StocksRepository:
    """Get the global stocks repository instance"""
    global _stocks_repo
    if _stocks_repo is None:
        _stocks_repo = StocksRepository()
    return _stocks_repo

