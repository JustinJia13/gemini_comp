"""
Download/log Gemini *hourly* crypto prediction markets data (BTC/ETH/SOL only)
and store under: .data/gemini/prediction_data/


Run:
  python gemini_hourly_predictions_logger.py

Optional:
  python gemini_hourly_predictions_logger.py --poll-sec 1 --refresh-sec 30 --book-levels 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE = "https://api.gemini.com"
OUT_ROOT = os.path.join(".data", "gemini", "prediction_data")

UNDERLYINGS = ("BTC", "ETH", "SOL")
#TODO: get data for gemini crypto prediction contracts. 