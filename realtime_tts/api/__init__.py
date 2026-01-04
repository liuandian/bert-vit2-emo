"""
API服务模块

提供HTTP REST API和WebSocket流式服务
"""

from .server import create_app, start_server

__all__ = ["create_app", "start_server"]
