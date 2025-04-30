import sqlite3
import threading
from typing import Optional, Dict


class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_table()

    def _create_table(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_usage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image_processing_time REAL NOT NULL,  -- Время обработки изображения (в секундах)
                    objects_count INTEGER NOT NULL,  -- Количество объектов на изображении
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP  -- Время создания записи
                )
                """
            )
            self.conn.commit()

    def add_record(
        self,
        user_id: int,
        image_processing_time: float,
        objects_count: int,
    ) -> None:
        """Добавляет запись о использовании бота в базу данных."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO bot_usage_history (
                    user_id,
                    image_processing_time,
                    objects_count
                ) VALUES (?, ?, ?)
                """,
                (user_id, image_processing_time, objects_count),
            )
            self.conn.commit()

    def get_user_history(self, user_id: int) -> Optional[Dict]:
        """Возвращает историю использования бота для конкретного пользователя."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM bot_usage_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                """,
                (user_id,),
            )
            return cursor.fetchall()

    def get_total_statistics(self) -> Dict[str, any]:
        """
        Возвращает общую статистику и статистику по пользователям:
        - total_processing_time: суммарное время обработки (сек)
        - total_requests: общее количество запросов
        - total_objects: общее количество объектов
        - users_stats: список с статистикой по каждому пользователю
        """
        with self.lock:
            cursor = self.conn.cursor()

            cursor.execute(
                """
                SELECT 
                    SUM(image_processing_time),
                    COUNT(*),
                    SUM(objects_count)
                FROM bot_usage_history
            """
            )
            total_stats = cursor.fetchone()

            cursor.execute(
                """
                SELECT 
                    user_id,
                    SUM(image_processing_time),
                    COUNT(*),
                    SUM(objects_count)
                FROM bot_usage_history
                GROUP BY user_id
                ORDER BY COUNT(*) DESC
            """
            )
            users_stats = [
                {
                    "user_id": row[0],
                    "processing_time": row[1],
                    "requests": row[2],
                    "objects": row[3],
                }
                for row in cursor.fetchall()
            ]

            return {
                "total_processing_time": total_stats[0] or 0.0,
                "total_requests": total_stats[1] or 0,
                "total_objects": total_stats[2] or 0,
                "users_stats": users_stats,
            }

    def close(self):
        """Закрывает соединение с базой данных."""
        self.conn.close()
