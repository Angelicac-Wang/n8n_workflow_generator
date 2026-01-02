#!/usr/bin/env python3
"""
日誌工具

提供統一的日誌記錄功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "n8n_workflow_recommender",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    設置日誌記錄器
    
    Args:
        name: 日誌記錄器名稱
        log_level: 日誌級別
        log_file: 日誌檔案路徑（可選）
    
    Returns:
        logger: 配置好的日誌記錄器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重複添加 handler
    if logger.handlers:
        return logger
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 檔案 handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

