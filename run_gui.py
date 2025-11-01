#!/usr/bin/env python3
"""Launch Speech Splitter GUI"""

import sys

def check_deps():
    missing = []
    try:
        import PySide6
    except ImportError:
        missing.append('PySide6')
    
    try:
        import whisper
    except ImportError:
        missing.append('openai-whisper')
    
    try:
        import pydub
    except ImportError:
        missing.append('pydub')
    
    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall: pip install -r requirements.txt")
        return False
    return True

if __name__ == '__main__':
    if check_deps():
        from speech_splitter_gui import main
        main()
    else:
        sys.exit(1)
