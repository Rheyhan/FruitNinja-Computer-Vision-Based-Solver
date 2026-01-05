'''
IN A NUTSHELL
CONCEPT OF THE SCRIPT:
1. Capture the screen indefinitely 800 x 600 px, if ur monitor is larger than that please change ur resolution first.
2. Use YOLOv11 to detect any object on each frame.
3. Use Deep SORT to track detected objects across frames.

TRACK LOGIC:
- If class 1 (E.g, bomb) is detected, prioritize a selective and safe tracking on class 0 (E.g, fruit) in order to minimize risk of collateral. In which, the farthest fruit from the bomb will be tracked and targeted first.
- If class 1 is not detected, track all detected class 0. And go crazy

USAGE:
Press 's' to start/pause the script.
Press 'q' to quit the script.

STILL WIP, THE OLD CODE SUCKS ASS!!!!!!!!!!!!!
'''