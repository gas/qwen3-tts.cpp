import os
import time
import subprocess
import requests
import argparse
import concurrent.futures
from urllib.parse import urljoin

SERVER_CMD = "./build/qwen3-tts-server"
MODEL_DIR = "./models"
TTS_MODEL = "qwen3-tts-1.7b-f16.gguf"
PORT = 8080
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"

def wait_for_server(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(urljoin(BASE_URL, "/health"))
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def run_inference(texts, name="Inference", is_concurrent=False):
    payload = {
        "input": texts,
        "language": "es"
    }
    
    start = time.time()
    try:
        resp = requests.post(urljoin(BASE_URL, "/v1/audio/generations"), json=payload)
        resp.raise_for_status()
        end = time.time()
        elapsed = end - start
        
        # Calculate TTFB (Time to First Byte) / Response time
        # For batched JSON, the whole response arrives at once, so Wall Time is TTFB + Transfer.
        if not is_concurrent:
            num_audios = len(texts) if isinstance(texts, list) else 1
            print(f"[{name}] Completed {num_audios} audios in {elapsed:.3f} seconds.")
        return elapsed
    except Exception as e:
        if not is_concurrent:
            print(f"[{name}] Failed: {e}")
        return -1.0

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS Server API")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of texts in the single batched request")
    parser.add_argument("--concurrency", type=int, default=6, help="Number of concurrent individual requests")
    args = parser.parse_args()

    if not os.path.exists(SERVER_CMD):
        print(f"Error: Server binary not found at {SERVER_CMD}. Please build it first.")
        return

    print("Starting Qwen3-TTS Server (Baseline Mode - No Voice Ref)...")
    # Launching server with a large enough internal chunk limit
    cmd = [SERVER_CMD, "-m", MODEL_DIR, "-tts", TTS_MODEL, "-p", str(PORT), "-b", str(args.batch_size)]
    
    server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        print("Waiting for server to load models...")
        if not wait_for_server(timeout=60):
            print("Error: Server failed to start.")
            return
            
        print("Server is ready!\n")
        
        test_txt = "Esta es una frase generica para medir tiempos."
        
        # 1. Cold Start Inference
        cold_time = run_inference(test_txt, "Cold Start (1 text)")
        
        # 2. Warm Start Inference
        time.sleep(1)
        warm_time = run_inference(test_txt, "Warm Start (1 text)")
        
        # 3. Batched Request (Single HTTP Request, Multiple Texts)
        time.sleep(1)
        print(f"\n--- Batched Inference (1 request with {args.batch_size} texts) ---")
        batch_payload = [f"Prueba de servidor en lote, frase numero {i}." for i in range(args.batch_size)]
        batch_time = run_inference(batch_payload, "Batched Request")
        
        # 4. Concurrent Requests (N HTTP Requests, 1 Text Each)
        time.sleep(1)
        print(f"\n--- Concurrent Network Requests ({args.concurrency} serialized by Mutex) ---")
        
        concurrent_times = []
        start_concurrent = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_inference, f"Peticion concurrente numero {i}.", f"Client {i}", True) for i in range(args.concurrency)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res > 0:
                    concurrent_times.append(res)
                    
        total_concurrent_time = time.time() - start_concurrent
        
        print("\n" + "="*60)
        print("SERVER TIMING COMPARISON")
        print("="*60)
        print(f"Single Request (Cold):        {cold_time:.3f} s")
        print(f"Single Request (Warm):        {warm_time:.3f} s")
        print("-" * 60)
        print(f"Batched Request ({args.batch_size} texts):    {batch_time:.3f} s total  [GPU Parallelized]")
        if args.batch_size > 0:
            print(f"  -> Average per audio:       {batch_time/args.batch_size:.3f} s")
        print("-" * 60)
        print(f"Concurrent Requests ({args.concurrency} txts): {total_concurrent_time:.3f} s total  [Mutex Serialized]")
        if args.concurrency > 0:
            print(f"  -> Average per audio:       {total_concurrent_time/args.concurrency:.3f} s")
        print("="*60)
            
    finally:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
