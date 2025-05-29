import os
import json
import time
from datetime import datetime
from pathlib import Path

class ModelOutputLogger:
    """
    用于记录模型输出的日志记录器，将每次训练和验证的输出保存到日志目录中
    """
    def __init__(self, log_dir="logs/model_outputs"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录的基础路径
        """
        self.base_log_dir = log_dir
        # 创建基础日志目录
        os.makedirs(self.base_log_dir, exist_ok=True)
        
        # 创建带时间戳的运行目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_log_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 创建训练和验证子目录
        self.train_dir = os.path.join(self.run_dir, "train")
        self.val_dir = os.path.join(self.run_dir, "validation")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        
        # 初始化计数器
        self.train_counter = 0
        self.val_counter = 0
        
        print(f"模型输出日志将保存到: {self.run_dir}")
    
    def log_train_sample(self, prompt, response, ground_truth, score, metadata=None):
        """
        记录训练样本的输出
        
        Args:
            prompt: 提示文本
            response: 模型响应
            ground_truth: 标准答案
            score: 评分
            metadata: 额外元数据
        """
        self._log_sample(
            dir_path=self.train_dir,
            counter=self.train_counter,
            prompt=prompt,
            response=response,
            ground_truth=ground_truth,
            score=score,
            metadata=metadata
        )
        self.train_counter += 1
    
    def log_val_sample(self, prompt, response, ground_truth, score, metadata=None):
        """
        记录验证样本的输出
        
        Args:
            prompt: 提示文本
            response: 模型响应
            ground_truth: 标准答案
            score: 评分
            metadata: 额外元数据
        """
        self._log_sample(
            dir_path=self.val_dir,
            counter=self.val_counter,
            prompt=prompt,
            response=response,
            ground_truth=ground_truth,
            score=score,
            metadata=metadata
        )
        self.val_counter += 1
    
    def _log_sample(self, dir_path, counter, prompt, response, ground_truth, score, metadata=None):
        """
        内部方法：记录单个样本，以朴素文本格式保存
        """
        # 创建带有计数器的文件名
        filename = f"sample_{counter:06d}.txt"
        filepath = os.path.join(dir_path, filename)
        
        # 使用朴素文本格式写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            # 添加标题和基本信息
            f.write("=" * 80 + "\n")
            f.write(f"样本ID: {counter:06d}\n")
            f.write(f"时间戳: {time.time()}\n")
            f.write(f"日期时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评分: {score}\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入提示文本
            f.write("【提示】\n")
            f.write("-" * 80 + "\n")
            f.write(prompt + "\n\n")
            
            # 写入模型响应
            f.write("【响应】\n")
            f.write("-" * 80 + "\n")
            f.write(response + "\n\n")
            
            # 写入标准答案（如果有）
            if ground_truth:
                f.write("【标准答案】\n")
                f.write("-" * 80 + "\n")
                f.write(ground_truth + "\n\n")
            
            # 写入元数据（如果有）
            if metadata:
                f.write("【元数据】\n")
                f.write("-" * 80 + "\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n") 