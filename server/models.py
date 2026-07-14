"""
数据库模型定义
使用SQLite存储用户、知识库、操作日志等结构化数据
"""
import os
import sqlite3
import hashlib
from datetime import datetime
from config import Config


class DatabaseManager:
    """数据库管理器，负责所有SQLite数据库操作"""

    def __init__(self):
        """初始化数据库连接并创建表结构"""
        Config.init_directories()
        self.conn = sqlite3.connect(Config.SQLITE_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
        self.create_tables()
        self.init_admin_user()

    def create_tables(self):
        """创建系统所需的全部数据表"""
        cursor = self.conn.cursor()

        # 用户表 - 存储用户基本信息和角色
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           username
                           TEXT
                           UNIQUE
                           NOT
                           NULL,
                           password_hash
                           TEXT
                           NOT
                           NULL,
                           role
                           TEXT
                           NOT
                           NULL
                           DEFAULT
                           'user', -- 'admin' 或 'user'
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           last_login
                           TIMESTAMP
                       )
                       ''')

        # 知识库表 - 记录上传的文档信息
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS knowledge_bases
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           name
                           TEXT
                           NOT
                           NULL,
                           description
                           TEXT,
                           file_path
                           TEXT,
                           file_type
                           TEXT,
                           chunk_count
                           INTEGER
                           DEFAULT
                           0,
                           created_by
                           INTEGER,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           created_by
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        # 操作日志表 - 记录所有用户操作
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS operation_logs
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER,
                           action
                           TEXT
                           NOT
                           NULL,
                           details
                           TEXT,
                           ip_address
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        # 问答统计表 - 记录问答使用情况
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS qa_stats
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER,
                           question
                           TEXT,
                           answer
                           TEXT,
                           response_time
                           REAL, -- 响应时间（秒）
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       )
                           )
                       ''')

        self.conn.commit()

    def init_admin_user(self):
        """初始化管理员账号，密码使用MD5加密"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (Config.ADMIN_USERNAME,))

        if not cursor.fetchone():
            cursor.execute('''
                           INSERT INTO users (username, password_hash, role)
                           VALUES (?, ?, ?)
                           ''', (Config.ADMIN_USERNAME, Config.ADMIN_PASSWORD_MD5, 'admin'))
            self.conn.commit()

    @staticmethod
    def hash_password(password):
        """使用MD5加密密码"""
        return hashlib.md5(password.encode('utf-8')).hexdigest()

    def verify_user(self, username, password):
        """验证用户登录信息"""
        cursor = self.conn.cursor()
        password_hash = self.hash_password(password)
        cursor.execute(
            'SELECT * FROM users WHERE username = ? AND password_hash = ?',
            (username, password_hash)
        )
        user = cursor.fetchone()

        if user:
            # 更新最后登录时间
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE id = ?',
                (datetime.now(), user['id'])
            )
            self.conn.commit()
            return dict(user)
        return None

    def add_user(self, username, password, role='user'):
        """添加新用户"""
        cursor = self.conn.cursor()
        try:
            password_hash = self.hash_password(password)
            cursor.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                (username, password_hash, role)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # 用户名已存在

    def get_all_users(self):
        """获取所有用户列表"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, username, role, created_at, last_login FROM users')
        return [dict(row) for row in cursor.fetchall()]

    def add_knowledge_base(self, name, description, file_path, file_type, chunk_count, created_by):
        """添加知识库记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       INSERT INTO knowledge_bases (name, description, file_path, file_type, chunk_count, created_by)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (name, description, file_path, file_type, chunk_count, created_by))
        self.conn.commit()
        return cursor.lastrowid

    def get_all_knowledge_bases(self):
        """获取所有知识库"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       SELECT kb.*, u.username as creator_name
                       FROM knowledge_bases kb
                                LEFT JOIN users u ON kb.created_by = u.id
                       ORDER BY kb.created_at DESC
                       ''')
        return [dict(row) for row in cursor.fetchall()]

    def add_log(self, user_id, action, details, ip_address='127.0.0.1'):
        """添加操作日志"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       INSERT INTO operation_logs (user_id, action, details, ip_address)
                       VALUES (?, ?, ?, ?)
                       ''', (user_id, action, details, ip_address))
        self.conn.commit()

    def add_qa_stat(self, user_id, question, answer, response_time):
        """添加问答统计"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       INSERT INTO qa_stats (user_id, question, answer, response_time)
                       VALUES (?, ?, ?, ?)
                       ''', (user_id, question, answer, response_time))
        self.conn.commit()

    def get_logs(self, limit=50):
        """获取最近的操作日志"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       SELECT ol.*, u.username
                       FROM operation_logs ol
                                LEFT JOIN users u ON ol.user_id = u.id
                       ORDER BY ol.created_at DESC LIMIT ?
                       ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self):
        """获取系统统计数据，用于管理后台图表展示"""
        cursor = self.conn.cursor()

        # 用户总数
        cursor.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']

        # 知识库总数
        cursor.execute('SELECT COUNT(*) as count FROM knowledge_bases')
        total_kb = cursor.fetchone()['count']

        # 今日问答数
        cursor.execute('''
                       SELECT COUNT(*) as count
                       FROM qa_stats
                       WHERE date (created_at) = date ('now')
                       ''')
        today_qa = cursor.fetchone()['count']

        # 总问答数
        cursor.execute('SELECT COUNT(*) as count FROM qa_stats')
        total_qa = cursor.fetchone()['count']

        # 最近7天每日问答数（用于图表）
        cursor.execute('''
                       SELECT date (created_at) as date, COUNT (*) as count
                       FROM qa_stats
                       WHERE created_at >= date ('now', '-7 days')
                       GROUP BY date (created_at)
                       ORDER BY date (created_at)
                       ''')
        daily_qa = [dict(row) for row in cursor.fetchall()]

        return {
            'total_users': total_users,
            'total_kb': total_kb,
            'today_qa': today_qa,
            'total_qa': total_qa,
            'daily_qa': daily_qa
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def delete_knowledge_base(self, kb_id: int) -> bool:
        """
        删除知识库记录

        Args:
            kb_id: 知识库ID
        Returns:
            bool: 是否成功
        """
        cursor = self.conn.cursor()
        try:
            # 先获取文件路径
            cursor.execute('SELECT file_path, name FROM knowledge_bases WHERE id = ?', (kb_id,))
            row = cursor.fetchone()

            if not row:
                return False

            file_path = row['file_path']

            # 删除数据库记录
            cursor.execute('DELETE FROM knowledge_bases WHERE id = ?', (kb_id,))
            self.conn.commit()

            # 删除物理文件
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

            return True
        except Exception as e:
            print(f"删除知识库失败: {str(e)}")
            return False

    def get_knowledge_base_by_id(self, kb_id: int) -> dict:
        """获取单个知识库信息"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM knowledge_bases WHERE id = ?', (kb_id,))
        row = cursor.fetchone()
        return dict(row) if row else None