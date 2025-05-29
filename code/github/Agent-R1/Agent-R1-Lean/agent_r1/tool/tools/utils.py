import re
import json
import typing
import pathlib
import datetime
import traceback
import subprocess

import datasets

class DataUtil:
    @staticmethod
    def load_from_dir(path: str) -> list[str]:
        test_dir = pathlib.Path(path)
        testcases = []
        for fp in test_dir.iterdir():
            testcases.append(fp.read_text(encoding='utf-8'))
        return testcases
    
    @staticmethod
    def load_from_data_parquet(path: str) -> list[str]:
        data = datasets.load_dataset('parquet', data_files=path, split='train')
        testcases = [fm['reward_model']['ground_truth']['formal_statement'] + ' sorry' for fm in data]
        return testcases
    
    @staticmethod
    def load_from_verification_log(path: str = '/volume/ailab4sci/users/lorm/logs/verification_log.jsonl', shuffle=True) -> list[str]:
        data = datasets.load_dataset('json', data_files=path, split='train')
        if shuffle:
            data = data.shuffle()
        data = data.map(lambda x: {'generated_formal_proof': x['formal_statement'].strip() + x['proof_code']})
        return data['generated_formal_proof']

class CodeUtil:
    @staticmethod
    def match_lean_code(solution_str: str) -> typing.Optional[str]:
        match1 = re.search(r"```lean4\s*([\s\S]*?)```", solution_str)
        if match1:
            return match1.group(1).strip('\n').strip()
        match2 = re.search(r"```lean\s*([\s\S]*?)```", solution_str)
        if match2:
            return match2.group(1).strip('\n').strip()
    
    @staticmethod
    def remove_comment(code: str) -> str:
        multiline_pattern = r'/-((!)?.*?)-/'
        code = re.sub(multiline_pattern, '', code, flags=re.DOTALL)
        singleline_pattern = r'--.*?$'
        # Using re.MULTILINE to make $ match the end of each line
        code = re.sub(singleline_pattern, '', code, flags=re.MULTILINE)
        return code
    
    @staticmethod
    def split_headers(code: str) -> typing.Optional[str]:
        pass

    @staticmethod
    def find_lean_theorem_end_pos(lean_code: str) -> int:
        """
        找到 Lean4 字符串中最后一个 theorem/lemma/instance/example 后紧跟的 `:= by` 或 `:=by` 的结束位置。
        
        参数:
            lean_code: 包含 Lean4 定理和证明的字符串
            
        返回:
            最后一个匹配的 `:= by` 或 `:=by` 的结束位置（字符索引，从 0 开始），
            如果没有找到则返回 -1。
        """
        # 匹配 theorem/lemma/instance/example 后跟着 := by 或 :=by
        pattern = r'(?:theorem|lemma|instance|example).*?:=\s*by'
        
        # 查找所有匹配项
        matches = list(re.finditer(pattern, lean_code, re.DOTALL))
        
        if not matches:
            return -1  # 没有找到匹配
        
        # 取最后一个匹配
        last_match = matches[-1]
        
        # 返回匹配的结束位置（即 := by 后的第一个字符的位置）
        return last_match.end()
    
    @staticmethod
    def split_lean_proof_code(code: str) -> typing.Optional[str]:
        if code is None:
            return None
        end_pos = CodeUtil.find_lean_theorem_end_pos(code)
        if end_pos == -1:
            return None
        else:
            return code[end_pos:]
        # if ":= by" in code:
        #     return code.split(":= by")[-1]
        # elif ":=by" in code:
        #     return code.split(":=by")[-1]
        # return None
    
    @staticmethod
    def extract_solution(solution_str: str, method='strict') -> typing.Optional[str]:
        """Extract the proof code from model output."""
        code = CodeUtil.match_lean_code(solution_str)
        if code is None:
            return None
        if method == 'strict':
            return CodeUtil.split_lean_proof_code(code)
        return code
    
    @staticmethod
    def verify_code_statement(code: typing.Optional[str], formal_statement: str) -> bool:
        if code is None:
            return False
        if 'import' not in code:
            return False
        if formal_statement not in code:
            return False
        return True
    
    @staticmethod
    def has_bonus(code: typing.Optional[str]) -> bool:
        if code is None:
            return False
        code = CodeUtil.remove_comment(code)
        if 'lemma' in code:
            return True
        if 'def' in code:
            return True
        return False
    
    @staticmethod
    def extract_last_statement_from_code(code: str, keywords: list[str]=['lemma', 'theorem', 'example']):
        pattern = rf'({"|".join(keywords)})'
        matches = list(re.finditer(pattern, code, re.IGNORECASE))
    
        if not matches:
            return code
    
        # Get the last match
        last_match = matches[-1]
        start_pos = last_match.start()
    
        # Return from the last keyword to the end of string
        return code[start_pos:]

class LogUtil:
    # 直接写入日志文件避免Ray日志前缀
    @staticmethod
    def write_log(message, path, level="INFO"):
        with open(path, 'a') as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"{timestamp} - {level} - {message}\n")

    @staticmethod
    def write_log_jsonl(message: typing.Any, path, level="INFO"):
        message = json.dumps(message, ensure_ascii=False, indent=4) + '\n\n'
        LogUtil.write_log(message, path, level)
    
    @staticmethod
    def log_verification_attempt(
        result: dict, 
        verification_log_path: str, 
        log_path: str
    ):
        """Log verification attempt to JSONL file."""
        try:
            with open(verification_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            LogUtil.write_log(f"Failed to log verification attempt: {e}", log_path, "ERROR")  

class ProcessUtil:
    @staticmethod
    def kill_repl() -> None:
        cmd = ['pkill', '-9', 'repl']
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
            print('Killed all repl process')
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                print('No running repl process to kill')
            else:
                print(traceback.format_exc())
    
    @staticmethod
    def kill_lean() -> None:
        cmd = ['pkill', '-9', 'lean']
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
            print('Killed all lean process')
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                print('No running lean process to kill')
            else:
                print(traceback.format_exc())