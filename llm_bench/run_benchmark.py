import yaml
import asyncio
import random
from ollama import AsyncClient
from datetime import datetime, timedelta
import subprocess
import threading
import time

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
            stream=False
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

# GPU monitoring functions
def get_gpu_info():
    """Retrieve GPU clock speed and power usage."""
    try:
        clock_speed_output = subprocess.check_output(['nvidia-smi', '--query-gpu=clocks.gr', '--format=csv,nounits'], text=True)
        power_usage_output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits'], text=True)
        clock_speed = clock_speed_output.split('\n')[1].strip()
        power_usage = power_usage_output.split('\n')[1].strip()
        return float(clock_speed), float(power_usage)
    except subprocess.CalledProcessError as e:
        print(f"Failed to retrieve GPU info: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error when retrieving GPU info: {e}")
        return None, None

gpu_data = []
stop_monitoring = threading.Event()

def monitor_gpu():
    global gpu_data
    while not stop_monitoring.is_set():
        try:
            clock_speed, power_usage = get_gpu_info()
            if clock_speed is not None and power_usage is not None:
                timestamp = datetime.now().isoformat()
                gpu_data.append({
                    'timestamp': timestamp,
                    'clock_speed': clock_speed,
                    'power_usage': power_usage
                })
            time.sleep(1)  # Collect data every minute
        except Exception as e:
            print(f"Error in GPU monitoring thread: {e}")
            time.sleep(1)  # Wait a bit before trying again

def get_gpu_stats(gpu_data, start_time, end_time):
    relevant_data = [d for d in gpu_data if start_time <= datetime.fromisoformat(d['timestamp']) <= end_time]
    if relevant_data:
        avg_clock_speed = sum(d['clock_speed'] for d in relevant_data) / len(relevant_data)
        avg_power_usage = sum(d['power_usage'] for d in relevant_data) / len(relevant_data)
        return avg_clock_speed, avg_power_usage
    return None, None

async def run_benchmark(models_file_path, steps, benchmark_file_path, is_test, ollamabin, concurrent, should_monitor_gpu, verbose):
    global gpu_data
    gpu_data = []
    monitoring_thread = None

    if should_monitor_gpu:
        try:
            # Test GPU info retrieval before starting the monitoring thread
            clock_speed, power_usage = get_gpu_info()
            if clock_speed is None or power_usage is None:
                print("Warning: Unable to retrieve GPU information. GPU monitoring will be disabled.")
                should_monitor_gpu = False
            else:
                monitoring_thread = threading.Thread(target=monitor_gpu)
                monitoring_thread.start()
                print("GPU monitoring started.")
        except Exception as e:
            print(f"Error setting up GPU monitoring: {e}")
            print("GPU monitoring will be disabled.")
            should_monitor_gpu = False

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
        print(f"Starting evaluation of {model_name}\n")

        if model_name in allowed_models:
            total_tokens = 0
            total_eval_duration = 0.0
            total_load_duration = 0.0
            total_prompt_eval_duration = 0.0
            total_duration = 0.0
            num_batches = 0

            model_start_time = datetime.now()
            model_gpu_data = []

            for step in range(num_steps):
                current_prompts = random.sample(prompt_dict, min(concurrent, len(prompt_dict)))
                tasks = []
                for index, prompt in enumerate(current_prompts, start=step * concurrent + 1):
                    prompt_text = prompt['prompt']
                    tasks.append(await add_query(model_name, prompt_text, index))

                task_results = await asyncio.gather(*tasks)

                # Filter out None results
                valid_results = [result for result in task_results if result is not None]

                if not valid_results:
                    print(f"Warning: No valid results for step {step + 1}. Skipping this step.")
                    continue

                step_eval_durations = []
                step_load_durations = []
                step_prompt_eval_durations = []

                first_start_time = min(valid_results, key=lambda x: x['load_duration'] + x['prompt_eval_duration'])
                first_start_time = first_start_time['load_duration'] + first_start_time['prompt_eval_duration']

                last_end_time = max(valid_results, key=lambda x: x['total_duration'])['total_duration']

                batch_eval_duration = last_end_time - first_start_time

                batch_tokens = sum(result['eval_count'] for result in valid_results)
                batch_avg_time_to_first_token = sum(result['load_duration'] + result['prompt_eval_duration'] for result in valid_results) / len(valid_results)

                print(f"\nEvaluating batch {step + 1}/{num_steps}\n")
                print("-----------------------------")
                print(f"Tokens per second: {batch_tokens / batch_eval_duration:.3f}")
                print(f"Produced tokens: {batch_tokens}")
                print(f"Total inference time: {batch_eval_duration:.3f}")
                print(f"Total seconds: {last_end_time:.3f}")
                print(f"Average time to first token: {batch_avg_time_to_first_token:.3f}")

                if should_monitor_gpu:
                    batch_start_time = model_start_time + timedelta(seconds=first_start_time)
                    batch_end_time = model_start_time + timedelta(seconds=last_end_time)
                    avg_clock_speed, avg_power_usage = get_gpu_stats(gpu_data, batch_start_time, batch_end_time)
                    if avg_clock_speed is not None and avg_power_usage is not None:
                        model_gpu_data.append({
                            'step': step + 1,
                            'avg_clock_speed': avg_clock_speed,
                            'avg_power_usage': avg_power_usage
                        })
                        print(f"Avg GPU Clock Speed: {avg_clock_speed:.2f} MHz")
                        print(f"Avg GPU Power Usage: {avg_power_usage:.2f} W")

                print("-----------------------------")

                for result in valid_results:
                    total_tokens += result['eval_count']
                    step_eval_durations.append(result['eval_duration'])
                    step_load_durations.append(result['load_duration'])
                    step_prompt_eval_durations.append(result['prompt_eval_duration'])

                total_eval_duration += batch_eval_duration
                total_load_duration += sum(step_load_durations) / len(step_load_durations)
                total_prompt_eval_duration += sum(step_prompt_eval_durations) / len(step_prompt_eval_durations)
                total_duration += last_end_time
                num_batches += 1

            if num_batches > 0:
                average_eval_rate = total_tokens / total_eval_duration
                average_load_duration = total_load_duration / num_batches
                average_prompt_eval_duration = total_prompt_eval_duration / num_batches
                average_time_to_first_token = average_load_duration + average_prompt_eval_duration

                results[model_name] = {
                    'average_tokens_per_second': f"{average_eval_rate:.3f}",
                    'total_tokens': total_tokens,
                    'total_inference_seconds': f"{total_eval_duration:.3f}",
                    'average_model_loading_seconds': f"{average_load_duration:.3f}",
                    'average_prompt_processing_seconds': f"{average_prompt_eval_duration:.3f}",
                    'average_time_to_first_token': f"{average_time_to_first_token:.3f}",
                    'total_seconds': f"{total_duration:.3f}",
                    'concurrent_users': concurrent
                }

                if model_gpu_data:
                    overall_avg_clock_speed = sum(d['avg_clock_speed'] for d in model_gpu_data) / len(model_gpu_data)
                    overall_avg_power_usage = sum(d['avg_power_usage'] for d in model_gpu_data) / len(model_gpu_data)
                    results[model_name]['average_gpu_clock_speed'] = f"{overall_avg_clock_speed:.2f}"
                    results[model_name]['average_gpu_power_usage'] = f"{overall_avg_power_usage:.2f}"

                if not is_test:
                    print(f"Results for {model_name}: {results[model_name]}")
                    print('-' * 10 + "\n")
            else:
                print(f"Warning: No valid batches completed for model {model_name}")

    if should_monitor_gpu and monitoring_thread:
        stop_monitoring.set()
        monitoring_thread.join()

        if verbose:
            print("\nGPU Monitoring Data:")
            print("--------------------")
            for data in gpu_data:
                print(f"Timestamp: {data['timestamp']}, Clock Speed: {data['clock_speed']} MHz, Power Usage: {data['power_usage']} W")

    if is_test:
        print("Test run successful.")

    return results