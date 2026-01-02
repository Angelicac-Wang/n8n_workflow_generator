#!/usr/bin/env python3
"""
檔案載入工具

提供統一的檔案載入介面，支援 JSON、YAML 等格式。
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(file_path: str) -> Dict:
    """
    載入 JSON 檔案
    
    Args:
        file_path: JSON 檔案路徑
    
    Returns:
        data: JSON 數據字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"檔案不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """
    保存數據到 JSON 檔案
    
    Args:
        data: 要保存的數據
        file_path: 輸出檔案路徑
        indent: JSON 縮排
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_yaml(file_path: str) -> Dict:
    """
    載入 YAML 檔案
    
    Args:
        file_path: YAML 檔案路徑
    
    Returns:
        data: YAML 數據字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"檔案不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: str):
    """
    保存數據到 YAML 檔案
    
    Args:
        data: 要保存的數據
        file_path: 輸出檔案路徑
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

