import argparse
import yaml
import subprocess
import datetime
import pkg_resources

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
            data=yaml.safe_load(stream)
            #print(d)
        except yaml.YAMLError as e:
            print(e)
    return data

def run_benchmark(models_file_path, benchmark_file_path, ollamabin):
    models_dict = parse_yaml(models_file_path)
    benchmark_dict = parse_yaml(benchmark_file_path)
    allowed_models = {e['model'] for e in models_dict['models']}
    ans = {}

    for model in models_dict['models']:
        model_name = model['model']
        if model_name in allowed_models:
            loc_dt = datetime.datetime.today()
            file_path = f'log_{loc_dt.strftime("%Y-%m-%d-%H%M%S")}.log'
            with open(file_path, "w", encoding='utf-8') as file:
                stored_nums = []
                for prompt in benchmark_dict['prompts']:
                    prompt_text = prompt['prompt']
                    print(f"prompt = {prompt_text}")
                    result = subprocess.run([ollamabin, 'run', model_name, prompt_text, '--verbose'], capture_output=True, text=True, check=True, encoding='utf-8')
                    std_err = result.stderr
                    file.write(std_err)
                    
                    for line in std_err.split('\n'):
                        if 'eval rate' in line and 'prompt' not in line:
                            print(line)
                            number = float(line[-20:-8])
                            stored_nums.append(number)

                print("-"*20) 
                if stored_nums:
                    average = sum(stored_nums) / len(stored_nums)
                    print("Average of eval rate: ", round(average,3), " tokens/s")
                    ans[model_name] = f"{round(average,3):.2f}"
                print("-"*40)
                file.write("\n"+"-"*40)

    return ans

if __name__ == "__main__": 
    args = parser.parse_args()
    if (args.models is not None) and (args.benchmark is not None):
        run_benchmark(args.models, args.benchmark)
        print('-'*40)
        
        