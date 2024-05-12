import argparse
import yaml
import subprocess
import datetime
import re

parser = argparse.ArgumentParser(
    prog="python3 check_models.py",
    description="Before running check_models.py, please make sure you installed ollama successfully \
        on macOS, Linux, or WSL2 on Windows. You can check the website: https://ollama.com",
    epilog="Author: Jason Chuang")

parser.add_argument("-v",
                    "--verbose",
                    action="store_true",
                    help="this program helps you check whether you have ollama benchmark models installed")

parser.add_argument("-m",
                    "--models",
                    type=str,
                    help="provide benchmark models YAML file path. ex. ../data/benchmark_models.yml")

parser.add_argument("-b",
                    "--benchmark",
                    type=str,
                    help="provide benchmark YAML file path. ex. ../data/benchmark1.yml")

def parse_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return data

def parse_duration(duration_str):
    """ Parse duration from strings in various formats (s, ms, m+s) to seconds as a float. """
    if 'ms' in duration_str:
        return float(duration_str.replace('ms', '')) / 1000.0
    elif 's' in duration_str:
        match = re.match(r'(?:(\d+)m)?(\d*\.?\d*)s', duration_str)
        if match:
            minutes = int(match.group(1)) if match.group(1) else 0
            seconds = float(match.group(2)) if match.group(2) else 0
            return 60 * minutes + seconds
        else:
            raise ValueError(f"Unsupported duration format: {duration_str}")
    else:
        raise ValueError("Unsupported duration format")

def run_benchmark(models_file_path, benchmark_file_path, ollamabin):
    models_dict = parse_yaml(models_file_path)
    benchmark_dict = parse_yaml(benchmark_file_path)
    allowed_models = {e['model'] for e in models_dict['models']}
    ans = {}

    for model in models_dict['models']:
        model_name = model['model']
        if model_name in allowed_models:
            loc_dt = datetime.datetime.today()
            stored_nums = []
            total_tokens = 0
            total_eval_duration = 0.0
            total_duration = 0.0

            print(f"Starting evaluation for model: {model_name}")
            for index, prompt in enumerate(benchmark_dict['prompts'], start=1):
                prompt_text = prompt['prompt']
                print(f"Evaluating prompt {index}/{len(benchmark_dict['prompts'])}")
                try:
                    result = subprocess.run(
                        [ollamabin, 'run', model_name, prompt_text, '--verbose'],
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding='utf-8'
                    )
                    std_err = result.stderr or ''
                    print("STDERR:", std_err)
                    
                    for line in std_err.split('\n'):
                        if line.lstrip().startswith("prompt"):
                            continue
                        if 'eval rate:' in line:
                            number = float(line.split()[-2])
                            stored_nums.append(number)
                        if 'eval count:' in line:
                            total_tokens += int(line.split()[-2])
                        if 'duration:' in line:
                            parts = line.split(':')[-1].strip()
                            duration = parse_duration(parts)
                            if 'eval duration:' in line:
                                total_eval_duration += duration
                            elif 'total duration:' in line:
                                total_duration += duration
                except subprocess.SubprocessError as e:
                    print(f"Error during subprocess execution: {e}")

            if stored_nums:
                average = sum(stored_nums) / len(stored_nums)
                print("Average of eval rate: ", round(average, 3), " tokens/s")
                ans[model_name] = {
                    'average_eval_rate': f"{round(average, 3):.2f} tokens/s",
                    'total_tokens': total_tokens,
                    'total_eval_duration': f"{total_eval_duration:.3f}s",
                    'total_duration': f"{total_duration:.3f}s"
                }
            print(f"Results for {model_name}: {ans[model_name]}")
            print('-' * 10 + "\n")

    return ans

if __name__ == "__main__": 
    args = parser.parse_args()
    if args.models and args.benchmark:
        run_benchmark(args.models, args.benchmark)
        print('-' * 40)
