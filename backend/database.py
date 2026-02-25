"""
Local SQLite Database for Boutique Try-On System
All data stored locally - no external connections
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from contextlib import contextmanager

# Database location - local only
DB_PATH = Path(__file__).parent / "data" / "boutique.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize database tables"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Clients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Garment categories - unrestricted for boutique use
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garment_categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                parent_id TEXT,
                sort_order INTEGER DEFAULT 0,
                FOREIGN KEY (parent_id) REFERENCES garment_categories(id)
            )
        """)

        # Garments catalog
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS garments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category_id TEXT,
                description TEXT,
                image_path TEXT NOT NULL,
                thumbnail_path TEXT,
                tags TEXT,
                price REAL,
                sku TEXT,
                in_stock INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (category_id) REFERENCES garment_categories(id)
            )
        """)

        # Client photos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_photos (
                id TEXT PRIMARY KEY,
                client_id TEXT,
                image_path TEXT NOT NULL,
                thumbnail_path TEXT,
                notes TEXT,
                measurements TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (client_id) REFERENCES clients(id)
            )
        """)

        # Try-on sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tryon_sessions (
                id TEXT PRIMARY KEY,
                client_id TEXT,
                client_photo_id TEXT,
                garment_id TEXT,
                status TEXT DEFAULT 'pending',
                prompt TEXT,
                negative_prompt TEXT,
                settings TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (client_id) REFERENCES clients(id),
                FOREIGN KEY (client_photo_id) REFERENCES client_photos(id),
                FOREIGN KEY (garment_id) REFERENCES garments(id)
            )
        """)

        # Generated results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tryon_results (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                thumbnail_path TEXT,
                seed INTEGER,
                generation_time REAL,
                model_used TEXT,
                is_favorite INTEGER DEFAULT 0,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES tryon_sessions(id)
            )
        """)

        # Masks (for reuse)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS masks (
                id TEXT PRIMARY KEY,
                client_photo_id TEXT NOT NULL,
                mask_path TEXT NOT NULL,
                mask_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (client_photo_id) REFERENCES client_photos(id)
            )
        """)

        # Settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)

        # Insert default categories for boutique
        default_categories = [
            ("cat_everyday", "Everyday Wear", "Regular clothing items", None, 1),
            ("cat_formal", "Formal Wear", "Dresses, suits, evening wear", None, 2),
            ("cat_swimwear", "Swimwear", "Bikinis, one-pieces, cover-ups", None, 3),
            ("cat_lingerie", "Lingerie", "Intimate apparel, bras, underwear", None, 4),
            ("cat_sheer", "Sheer & Transparent", "See-through fabrics, mesh", None, 5),
            ("cat_fetish", "Specialty & Fetish", "Latex, leather, specialty items", None, 6),
            ("cat_bridal", "Bridal", "Wedding dresses, bridal lingerie", None, 7),
            ("cat_sleepwear", "Sleepwear", "Nightgowns, robes, pajamas", None, 8),
            ("cat_activewear", "Activewear", "Sports bras, leggings, gym wear", None, 9),
            ("cat_custom", "Custom Orders", "Bespoke and custom items", None, 10),
        ]

        for cat in default_categories:
            cursor.execute("""
                INSERT OR IGNORE INTO garment_categories (id, name, description, parent_id, sort_order)
                VALUES (?, ?, ?, ?, ?)
            """, cat)

        # Default settings
        default_settings = [
            ("default_steps", "25"),
            ("default_cfg", "7.0"),
            ("default_sampler", "euler_a"),
            ("default_resolution", "512x768"),
            ("watermark_enabled", "false"),
            ("auto_save_results", "true"),
            ("preferred_model", "realisticVision"),
        ]

        for key, value in default_settings:
            cursor.execute("""
                INSERT OR IGNORE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))


# ============== CRUD Operations ==============

class ClientDB:
    @staticmethod
    def create(id: str, name: str, email: str = None, phone: str = None, notes: str = None) -> Dict:
        now = datetime.now().isoformat()
        with get_db() as conn:
            conn.execute("""
                INSERT INTO clients (id, name, email, phone, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (id, name, email, phone, notes, now, now))
        return {"id": id, "name": name, "created_at": now}

    @staticmethod
    def get(id: str) -> Optional[Dict]:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM clients WHERE id = ?", (id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_all() -> List[Dict]:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM clients ORDER BY name").fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def update(id: str, **kwargs) -> bool:
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [id]
        with get_db() as conn:
            conn.execute(f"UPDATE clients SET {sets} WHERE id = ?", values)
        return True

    @staticmethod
    def delete(id: str) -> bool:
        with get_db() as conn:
            conn.execute("DELETE FROM clients WHERE id = ?", (id,))
        return True


class GarmentDB:
    @staticmethod
    def create(id: str, name: str, image_path: str, category_id: str = None, **kwargs) -> Dict:
        now = datetime.now().isoformat()
        tags = json.dumps(kwargs.get("tags", [])) if kwargs.get("tags") else None
        with get_db() as conn:
            conn.execute("""
                INSERT INTO garments (id, name, category_id, description, image_path, thumbnail_path,
                                      tags, price, sku, in_stock, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (id, name, category_id, kwargs.get("description"), image_path,
                  kwargs.get("thumbnail_path"), tags, kwargs.get("price"),
                  kwargs.get("sku"), kwargs.get("in_stock", 1), now, now))
        return {"id": id, "name": name, "image_path": image_path}

    @staticmethod
    def get(id: str) -> Optional[Dict]:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM garments WHERE id = ?", (id,)).fetchone()
            if row:
                result = dict(row)
                if result.get("tags"):
                    result["tags"] = json.loads(result["tags"])
                return result
            return None

    @staticmethod
    def list_by_category(category_id: str = None) -> List[Dict]:
        with get_db() as conn:
            if category_id:
                rows = conn.execute(
                    "SELECT * FROM garments WHERE category_id = ? ORDER BY name",
                    (category_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM garments ORDER BY name").fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def search(query: str) -> List[Dict]:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT * FROM garments
                WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                ORDER BY name
            """, (f"%{query}%", f"%{query}%", f"%{query}%")).fetchall()
            return [dict(row) for row in rows]


class TryOnSessionDB:
    @staticmethod
    def create(id: str, client_photo_id: str, garment_id: str = None,
               client_id: str = None, prompt: str = None, settings: dict = None) -> Dict:
        now = datetime.now().isoformat()
        settings_json = json.dumps(settings) if settings else None
        with get_db() as conn:
            conn.execute("""
                INSERT INTO tryon_sessions (id, client_id, client_photo_id, garment_id,
                                            status, prompt, settings, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            """, (id, client_id, client_photo_id, garment_id, prompt, settings_json, now, now))
        return {"id": id, "status": "pending", "created_at": now}

    @staticmethod
    def update_status(id: str, status: str) -> bool:
        now = datetime.now().isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE tryon_sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, id)
            )
        return True

    @staticmethod
    def get(id: str) -> Optional[Dict]:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM tryon_sessions WHERE id = ?", (id,)).fetchone()
            if row:
                result = dict(row)
                if result.get("settings"):
                    result["settings"] = json.loads(result["settings"])
                return result
            return None

    @staticmethod
    def list_recent(limit: int = 50) -> List[Dict]:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT * FROM tryon_sessions ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]


class TryOnResultDB:
    @staticmethod
    def create(id: str, session_id: str, image_path: str, **kwargs) -> Dict:
        now = datetime.now().isoformat()
        with get_db() as conn:
            conn.execute("""
                INSERT INTO tryon_results (id, session_id, image_path, thumbnail_path,
                                           seed, generation_time, model_used, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (id, session_id, image_path, kwargs.get("thumbnail_path"),
                  kwargs.get("seed"), kwargs.get("generation_time"),
                  kwargs.get("model_used"), kwargs.get("notes"), now))
        return {"id": id, "session_id": session_id, "image_path": image_path}

    @staticmethod
    def get_by_session(session_id: str) -> List[Dict]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM tryon_results WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def set_favorite(id: str, is_favorite: bool) -> bool:
        with get_db() as conn:
            conn.execute(
                "UPDATE tryon_results SET is_favorite = ? WHERE id = ?",
                (1 if is_favorite else 0, id)
            )
        return True

    @staticmethod
    def get_favorites() -> List[Dict]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM tryon_results WHERE is_favorite = 1 ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]


class SettingsDB:
    @staticmethod
    def get(key: str, default: str = None) -> Optional[str]:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    @staticmethod
    def set(key: str, value: str) -> bool:
        now = datetime.now().isoformat()
        with get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, now))
        return True

    @staticmethod
    def get_all() -> Dict[str, str]:
        with get_db() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            return {row["key"]: row["value"] for row in rows}


# Initialize database on import
init_database()
