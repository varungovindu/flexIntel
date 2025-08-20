#!/usr/bin/env python3
"""
Test script to demonstrate fullscreen and mirror fixes
"""

import cv2
import numpy as np

def test_fullscreen():
    """Test fullscreen window functionality"""
    print("🖥️ Testing Fullscreen Functionality")
    print("=" * 40)
    
    # Create a test window
    window_name = "flexIntel - Fullscreen Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create a test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "flexIntel Fullscreen Test", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 212, 170), 2)
    cv2.putText(img, "Press F to toggle fullscreen", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Press Q to quit", (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print("✅ Window created successfully")
    print("✅ Press F to toggle fullscreen mode")
    print("✅ Press Q to quit the test")
    
    # Set initial fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("✅ Fullscreen mode enabled by default")
    
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Toggle fullscreen
            current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            new_state = cv2.WINDOW_NORMAL if current_state == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_state)
            
            if new_state == cv2.WINDOW_FULLSCREEN:
                print("🖥️ Fullscreen mode enabled")
            else:
                print("📱 Windowed mode enabled")
    
    cv2.destroyAllWindows()
    print("✅ Test completed successfully")

if __name__ == "__main__":
    test_fullscreen()
