"""
认证授权模块
提供用户登录、注册、权限验证等功能
"""
from functools import wraps
from flask import request, jsonify, session
from models import DatabaseManager

# 全局数据库管理器实例
db_manager = DatabaseManager()


def login_required(f):
    """登录验证装饰器 - 确保用户已登录才能访问接口"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': '请先登录'}), 401
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """管理员权限验证装饰器 - 确保只有管理员能访问"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': '请先登录'}), 401
        if session.get('role') != 'admin':
            return jsonify({'error': '需要管理员权限'}), 403
        return f(*args, **kwargs)

    return decorated_function


def login_user(username, password):
    """用户登录函数"""
    user = db_manager.verify_user(username, password)
    if user:
        # 设置session信息
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']

        # 记录登录日志
        db_manager.add_log(
            user_id=user['id'],
            action='登录系统',
            details=f"用户 {username} 登录成功",
            ip_address=request.remote_addr
        )
        return True
    return False


def register_user(username, password, role='user'):
    """用户注册函数"""
    user_id = db_manager.add_user(username, password, role)
    if user_id:
        # 记录注册日志
        db_manager.add_log(
            user_id=user_id,
            action='用户注册',
            details=f"新用户 {username} 注册成功，角色: {role}",
            ip_address=request.remote_addr
        )
        return True
    return False


def logout_user():
    """用户登出函数"""
    if 'user_id' in session:
        # 记录登出日志
        db_manager.add_log(
            user_id=session['user_id'],
            action='退出系统',
            details=f"用户 {session.get('username')} 退出登录",
            ip_address=request.remote_addr
        )
        session.clear()