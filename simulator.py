"""
Aviation Data Simulator
=======================
Simulates an external engine system (like a FADEC or Ground Station)
sending real-time telemetry to our SARN Live Dashboard over HTTP.

Usage:
  1. Start the REST API receiver:
     python realtime_dashboard.py --source rest --port 5001
  
  2. In another terminal, run this simulator:
     python simulator.py --target http://localhost:5001/reading --engines 2
"""

import time
import json
import argparse
import random
import requests
import numpy as np
import threading
import sys

# Sensor list expected by the model
FEATURE_COLS = [
    "op1", "op2", "op3", 
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", 
    "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# Rough baseline "healthy" normalized values (approximate means)
HEALTHY_BASE = {
    "op1": 0.5, "op2": 0.5, "op3": 0.0,
    "s2": 0.3, "s3": 0.4, "s4": 0.35, "s7": 0.6, "s8": 0.5, "s9": 0.3,
    "s11": 0.4, "s12": 0.6, "s13": 0.5, "s14": 0.4, "s15": 0.4, 
    "s17": 0.4, "s20": 0.5, "s21": 0.5
}

class EngineSimulator:
    def __init__(self, engine_id):
        self.engine_id = engine_id
        self.cycle = 0
        self.degradation_factor = random.uniform(0.000001, 0.000005)
        self.anomaly_active = False
        self.anomaly_sensor = None
        self.anomaly_severity = 0.0
        
        # State holds current sensor values
        self.state = HEALTHY_BASE.copy()

    def step(self):
        self.cycle += 1
        
        # 1. Normal wear & tear (slow drift across all sensors)
        for s in FEATURE_COLS:
            noise = random.gauss(0, 0.0005) # normal sensor noise
            # some go up with age, some go down
            direction = 1 if s in ["s3", "s4", "s8", "s11", "s13", "s15", "s17"] else -1
            
            self.state[s] += (direction * self.degradation_factor) + noise
            self.state[s] = np.clip(self.state[s], 0.0, 1.0)
            
        # 2. Inject active anomaly if triggered
        if self.anomaly_active:
            # Huge spike or severe drift in one sensor
            self.state[self.anomaly_sensor] += self.anomaly_severity
            self.state[self.anomaly_sensor] = np.clip(self.state[self.anomaly_sensor], 0.0, 1.0)
            
        return {
            "engine_id": self.engine_id,
            "cycle": self.cycle,
            "timestamp": time.time(),
            "features": self.state.copy()
        }

    def trigger_anomaly(self):
        self.anomaly_active = True
        self.anomaly_sensor = random.choice(["s2", "s3", "s4", "s9", "s14"]) # Common failure sensors
        self.anomaly_severity = random.choice([0.15, -0.15, 0.25, -0.25])
        print(f"\n[WARNING] SIMULATOR: Triggered violent anomaly on Engine #{self.engine_id} in sensor {self.anomaly_sensor}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="http://localhost:5001/reading")
    parser.add_argument("--engines", type=int, default=1, help="Number of engines to simulate concurrently")
    parser.add_argument("--hz", type=float, default=5.0, help="Updates per second per engine")
    args = parser.parse_args()

    engines = [EngineSimulator(engine_id=i+1) for i in range(args.engines)]
    delay = 1.0 / args.hz

    print(f"==================================================")
    print(f" SARN External Telemetry Simulator")
    print(f" Target  : {args.target}")
    print(f" Engines : {args.engines}")
    print(f" Rate    : {args.hz} Hz")
    print(f"==================================================")
    print(f"Press 'a' + ENTER at any time to inject an anomaly into Engine #1")
    print(f"Press CTRL+C to quit.\n")

    # Keyboard listener to inject anomalies manually
    def listen_for_anomaly():
        while True:
            char = sys.stdin.read(1)
            if char.lower() == 'a':
                engines[0].trigger_anomaly()
                
    threading.Thread(target=listen_for_anomaly, daemon=True).start()

    session = requests.Session()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=args.engines * 2)

    def send_request(eng, payload):
        try:
            res = session.post(args.target, json=payload, timeout=0.5)
            if res.status_code == 200:
                sys.stdout.write(f"\r[Sent] Engine #{eng.engine_id} Cycle {eng.cycle}  -> OK     ")
            else:
                sys.stdout.write(f"\r[Sent] Engine #{eng.engine_id} Cycle {eng.cycle}  -> ERR: {res.status_code}")
        except requests.exceptions.RequestException as e:
            sys.stdout.write(f"\r[Error] Connection refused by {args.target}        ")
        sys.stdout.flush()

    try:
        while True:
            start_time = time.time()
            for eng in engines:
                payload = eng.step()
                executor.submit(send_request, eng, payload)
                
            elapsed = time.time() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
