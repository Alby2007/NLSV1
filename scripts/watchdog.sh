#!/bin/bash
while true; do
  if ! pgrep -f translate_dataset > /dev/null; then
    echo "$(date) restarting translator" >> /tmp/watchdog.log
    cd /workspace/NLSV1
    nohup python phase4_corpus/translate_dataset.py >> /tmp/translate.log 2>&1 &
  fi
  if ! pgrep -f "ollama serve" > /dev/null; then
    echo "$(date) restarting ollama" >> /tmp/watchdog.log
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 10
  fi
  sleep 30
done
