#!/usr/bin/env python3
"""Simple client to send a WAV file to the FastAPI cough detection server.

Usage:
  python send_wav.py --wav path/to/file.wav --url http://localhost:8000/infer --threshold 0.6
"""
import argparse
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True, help='Path to WAV file to send')
    ap.add_argument('--url', default='http://localhost:8000/infer', help='Inference endpoint URL')
    ap.add_argument('--threshold', type=float, default=0.6, help='Decision threshold')
    args = ap.parse_args()

    with open(args.wav, 'rb') as f:
        files = {'file': (args.wav, f, 'audio/wav')}
        params = {'threshold': args.threshold}
        r = requests.post(args.url, files=files, params=params, timeout=60)
        r.raise_for_status()
        print(r.json())


if __name__ == '__main__':
    main()
