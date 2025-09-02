import yaml
import os

def load_config(config_path="config/settings.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config