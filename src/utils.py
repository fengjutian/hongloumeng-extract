import yaml
import os
import time
import requests

def load_config(config_path="config/settings.yaml"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

def get_ollama_base_url(config):
    """从配置中获取Ollama基础URL"""
    return config.get("ollama_base_url", "http://localhost:11434")

def check_ollama_connection(base_url, timeout=5):
    """检查Ollama服务是否可达"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def wait_for_ollama_service(base_url, max_wait=30):
    """等待Ollama服务启动"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_ollama_connection(base_url):
            return True
        print("等待Ollama服务启动...")
        time.sleep(2)
    return False