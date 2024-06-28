import argparse
import yaml
import asyncio
import random
from ollama import AsyncClient

parser = argparse.ArgumentParser(
    prog="python3 check_models.py",
    description="Before running check_models.py, please make sure you installed ollama successfully \
        on macOS, Linux, or WSL2 on Windows. You can check the website: https://ollama.com")

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

parser.add_argument("-t",
                    "--test",
                    action="store_true",
                    help="run in test mode with minimal output")

parser.add_argument("-c",
                    "--concurrent",
                    type=int,
                    default=4,
                    help="number of concurrent prompts to process")

def parse_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return data

def ns_to_seconds(duration_ns):
    """ Convert nanoseconds to seconds. """
    return duration_ns / 1e9

async def run_query_async(model, query, instance_id):
    try:
        client = AsyncClient()
        response = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': query}],
            stream=False,
            options={'num_predict': 10}
        )

        # Extracting required metrics from response
        eval_count = response['eval_count']
        eval_duration = ns_to_seconds(response['eval_duration'])
        load_duration = ns_to_seconds(response['load_duration'])
        prompt_eval_duration = ns_to_seconds(response['prompt_eval_duration'])
        total_duration = ns_to_seconds(response['total_duration'])

        return {
            'eval_count': eval_count,
            'eval_duration': eval_duration,
            'load_duration': load_duration,
            'prompt_eval_duration': prompt_eval_duration,
            'total_duration': total_duration
        }
    except Exception as e:
        print(f"\nInstance {instance_id}: Error - {str(e)}")
        return None

async def add_query(model, query, instance_id):
    task = asyncio.create_task(run_query_async(model, query, instance_id))
    return task

async def run_benchmark(models_file_path, steps, benchmark_file_path, is_test, concurrent):
    models_dict = parse_yaml(models_file_path)
    benchmark_dict = parse_yaml(benchmark_file_path)
    allowed_models = {e['model'] for e in models_dict['models']}
    results = {}

    num_steps = steps
    prompt_dict = benchmark_dict['prompts']

    if is_test:
        prompt_dict = benchmark_dict["testPrompts"]
        num_steps = min(len(prompt_dict), steps)

    for model in models_dict['models']:
        model_name = model['model']
        if model_name in allowed_models:
            total_tokens = 0
            total_eval_duration = 0.0
            total_load_duration = 0.0
            total_prompt_eval_duration = 0.0
            total_duration = 0.0
            num_batches = 0

            for step in range(num_steps):
                current_prompts = random.sample(prompt_dict, min(concurrent, len(prompt_dict)))
                tasks = []
                for index, prompt in enumerate(current_prompts, start=step * concurrent + 1):
                    prompt_text = prompt['prompt']
                    tasks.append(await add_query(model_name, prompt_text, index))

                task_results = await asyncio.gather(*tasks)

                if not task_results:
                    continue

                step_eval_durations = []
                step_load_durations = []
                step_prompt_eval_durations = []

                first_start_time = min(task_results, key=lambda x: x['load_duration'] + x['prompt_eval_duration'])
                first_start_time = first_start_time['load_duration'] + first_start_time['prompt_eval_duration']

                last_end_time = max(task_results, key=lambda x: x['total_duration'])['total_duration']

                batch_eval_duration = last_end_time - first_start_time

                for result in task_results:
                    if result:
                        total_tokens += result['eval_count']
                        step_eval_durations.append(result['eval_duration'])
                        step_load_durations.append(result['load_duration'])
                        step_prompt_eval_durations.append(result['prompt_eval_duration'])

                if step_eval_durations:
                    total_eval_duration += batch_eval_duration
                    total_load_duration += sum(step_load_durations) / len(step_load_durations)
                    total_prompt_eval_duration += sum(step_prompt_eval_durations) / len(step_prompt_eval_durations)
                    total_duration += last_end_time
                    num_batches += 1

            average_eval_rate = total_tokens / total_eval_duration
            average_load_duration = total_load_duration / num_batches if num_batches else 0.0
            average_prompt_eval_duration = total_prompt_eval_duration / num_batches if num_batches else 0.0
            average_time_to_first_token = average_load_duration + average_prompt_eval_duration

            results[model_name] = {
                'average_token_per_second': f"{average_eval_rate:.3f}",
                'total_tokens': total_tokens,
                'total_inference_seconds': f"{total_eval_duration:.3f}",
                'average_model_loading_seconds': f"{average_load_duration:.3f}",
                'average_prompt_processing_seconds': f"{average_prompt_eval_duration:.3f}",
                'average_time_to_first_token': f"{average_time_to_first_token:.3f}",
                'total_seconds': f"{total_duration:.3f}",
            }
            if not is_test:
                print(f"Results for {model_name}: {results[model_name]}")
                print('-' * 10 + "\n")

    if is_test:
        print("Test run successful.")

    return results

if __name__ == "__main__": 
    args = parser.parse_args()
    if args.models and args.benchmark:
        asyncio.run(run_benchmark(args.models, 3, args.benchmark, args.test, args.concurrent))
        if not args.test:
            print('-' * 40)
