import os
import time
import subprocess
import re

CLI_PATH = "./build/qwen3-tts-cli"
MODEL_DIR = "./models"
TTS_MODEL = "qwen3-tts-1.7b-f16.gguf"
REF_AUDIO = "./audioref/voz_test_0493.q3vp"
REF_TEXT = "./audioref/voz_test_0493.txt"
TEST_TEXT = "Hola, esta es una prueba automatizada para medir la velocidad de inferencia de cada modalidad."
OUT_PREFIX = "./output/benchmark"

def run_command(cmd, name):
    print(f"\nEvaluating: {name}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {name}:\n{e.stderr}")
        return None
        
    end_time = time.time()
    
    # Extract metrics from stderr
    stderr = result.stderr
    
    # Try to find memory peak
    rss_peak_match = re.search(r"RSS peak:\s+([\d.]+ \w+)", stderr)
    rss_peak = rss_peak_match.group(1) if rss_peak_match else "N/A"
    
    # Find timing stats
    load_match = re.search(r"Load:\s+(\d+) ms", stderr)
    load_time = load_match.group(1) if load_match else "N/A"
    
    gen_match = re.search(r"Generate:\s+(\d+) ms", stderr)
    gen_time = gen_match.group(1) if gen_match else "N/A"
    
    rtf_match = re.search(r"RTF=([\d.]+)", stderr)
    rtf = rtf_match.group(1) if rtf_match else "N/A"
    
    ttft_match = re.search(r"Prefill \(.*?\):\s+([\d.]+) ms", stderr)
    ttft = ttft_match.group(1) if ttft_match else "N/A"
    if ttft == "N/A":
        # Alternative extraction for TTFT (graph build + prefill)
        pass # Not totally standard, RTF and Generation time are better
        
    print(f"  Total Time: {end_time - start_time:.2f}s")
    print(f"  Memory Peak: {rss_peak}")
    print(f"  Load Time: {load_time} ms")
    print(f"  Generation: {gen_time} ms")
    print(f"  RTF (Real Time Factor): {rtf}")
    
    return {
        "name": name,
        "rss": rss_peak,
        "load": load_time,
        "gen": gen_time,
        "rtf": rtf
    }

def main():
    if not os.path.exists(CLI_PATH):
        print(f"Error: {CLI_PATH} not found.")
        return
        
    results = []
    
    # 1. Zero-shot baseline
    cmd1 = [CLI_PATH, "-m", MODEL_DIR, "-tts", TTS_MODEL, "-t", TEST_TEXT, "-o", f"{OUT_PREFIX}_zeroshot.wav", "-l", "es"]
    res1 = run_command(cmd1, "Zero-shot Baseline")
    if res1: results.append(res1)
    
    # 2. X-Vector Only
    cmd2 = [CLI_PATH, "-m", MODEL_DIR, "-tts", TTS_MODEL, "-t", TEST_TEXT, "-r", REF_AUDIO, "-x", "-o", f"{OUT_PREFIX}_xvector.wav", "-l", "es"]
    res2 = run_command(cmd2, "X-Vector Only Clone")
    if res2: results.append(res2)
    
    # 3. Full ICL
    cmd3 = [CLI_PATH, "-m", MODEL_DIR, "-tts", TTS_MODEL, "-t", TEST_TEXT, "-r", REF_AUDIO, "-p", REF_TEXT, "-o", f"{OUT_PREFIX}_icl.wav", "-l", "es"]
    res3 = run_command(cmd3, "Full ICL Clone")
    if res3: results.append(res3)
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"{'Modality':<25} | {'Gen Time (ms)':<15} | {'RTF':<10} | {'Peak Mem'}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['gen']:<15} | {r['rtf']:<10} | {r['rss']}")

if __name__ == "__main__":
    main()
