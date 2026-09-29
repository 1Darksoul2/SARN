import subprocess
import time
import sys
import os

def main():
    print("==================================================")
    print(" SARN Master Runner")
    print("==================================================")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["GEMINI_API_KEY"] = ""
    
    # Step 1: Run pipeline evaluation (uses saved model)
    print("\n[1/3] Running evaluation pipeline (loading saved models)...")
    try:
        subprocess.run(
            [sys.executable, "run_pipeline.py", "--load_model", "--subset", "FD001"],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)
        
    # Step 2: Start the real-time dashboard in the background
    print("\n[2/3] Starting real-time dashboard on port 5001...")
    dashboard_process = subprocess.Popen(
        [sys.executable, "realtime_dashboard.py", "--source", "rest", "--port", "5001"],
        env=env
    )
    
    # Wait for dashboard to spin up
    print("Waiting for dashboard to initialize...")
    time.sleep(4)
    
    # Check if dashboard crashed immediately
    if dashboard_process.poll() is not None:
        print("Dashboard failed to start.")
        sys.exit(1)
        
    # Step 3: Start the data simulator
    print("\n[3/3] Starting real-time data simulator...")
    try:
        subprocess.run(
            [sys.executable, "simulator.py", "--target", "http://localhost:5001/reading", "--engines", "2"],
            env=env
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    finally:
        print("Terminating dashboard process...")
        dashboard_process.terminate()
        try:
            dashboard_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            dashboard_process.kill()
        print("All processes stopped.")

if __name__ == "__main__":
    main()
