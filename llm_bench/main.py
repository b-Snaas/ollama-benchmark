import typer
import pkg_resources
import speedtest
import time
from .systeminfo import sysmain
from .security_connection import connection 
from llm_bench import check_models, check_ollama, run_benchmark

app = typer.Typer()

@app.command(name="get_model_path")
def get_model_path(model: str) -> str:
    """ Helper function to return the correct file path based on the model size """
    return pkg_resources.resource_filename('llm_bench', f'data/{model}_models.yml')

@app.command(name="sysinfo")
def sysinfo(formal: bool = True):
    if formal:
        sys_info = sysmain.get_extra()
        sys_info['uuid'] = f"{sysmain.get_uuid()}"
        print(f"memory : {sys_info['memory']:.2f} GB") 
        print(f"cpu_info: {sys_info['cpu']}")
        print(f"gpu_info: {sys_info['gpu']}")
        print(f"os_version: {sys_info['os_version']}")
        print(f"Your machine UUID : {sys_info['uuid']}")

        x = connection.send_sysinfo(sys_info)
        print(x)
    else:
        print(f"No print!")

@app.command(name="check_internet_speed")
def check_internet_speed():
    start_time = time.time()
    st = speedtest.Speedtest()
    st.get_best_server()
    download_speed = st.download() / 1_000_000
    upload_speed = st.upload() / 1_000_000
    elapsed_time = time.time() - start_time
    print(f"Download Speed: {download_speed:.2f} Mbps")
    print(f"Upload Speed: {upload_speed:.2f} Mbps")
    print(f"Internet speed test duration: {elapsed_time:.2f} seconds")

@app.command(name="run")
def run(
    ollamabin: str = typer.Option('ollama', help="Path to the Ollama binary."),
    sendinfo: bool = typer.Option(False, help="Flag to send system info."),
    model: str = typer.Option(
        "all", help="Select model: all-small, gemma-small, llama3-small, mistral-small, phi3-small, qwen-small, all-medium, gemma-medium, phi3-medium, llama3-medium, all-large, drbx-large, llama3-large",
        case_sensitive=False, show_choices=True
    ),
    test: bool = typer.Option(
        False, '--test', help="Flag to toggle between default and alternative benchmark tests."
    ),
    steps: int = typer.Option(
        10, help="Set the amount of inferences",
        min=1, max=10
    )
):
    benchmark_file = pkg_resources.resource_filename('llm_bench', 'data/benchmark_instructions.yml')
    models_file = get_model_path(model)

    if sendinfo:
        sysmain.get_extra()

    ollama_version = check_ollama.check_ollama_version(ollamabin)
    print(f"ollama_version: {ollama_version} \n")

    start_time = time.time()
    check_models.pull_models(models_file)
    elapsed_time = time.time() - start_time
    print(f"Model pulling time: {elapsed_time:.2f} seconds")
    print('-' * 10)

    bench_results_info = {}
    result = run_benchmark.run_benchmark(models_file, steps, benchmark_file, test, ollamabin)
    bench_results_info.update(result)

    print(bench_results_info)

if __name__ == "__main__":
    app()
