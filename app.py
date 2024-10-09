import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize game variables
ball_pos = None
score = 0
lives = 5
game_over = False

def reset_game():
    global ball_pos, score, lives, game_over
    ball_pos = None
    score = 0
    lives = 5
    game_over = False

def create_ball():
    return (random.randint(50, 590), 0)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if not game_over:
        # Create a new ball if there isn't one
        if ball_pos is None:
            ball_pos = create_ball()

        # Move the ball down
        ball_pos = (ball_pos[0], ball_pos[1] + 5)

        # Draw the ball
        cv2.circle(frame, ball_pos, 20, (0, 0, 255), -1)

        # Check if the ball reached the bottom
        if ball_pos[1] >= 480:
            lives -= 1
            ball_pos = None
            if lives == 0:
                game_over = True

        # Check for hand collision
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the index finger tip
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

                # Check for collision with the ball
                if ball_pos is not None:
                    distance = np.sqrt((index_tip_x - ball_pos[0])**2 + (index_tip_y - ball_pos[1])**2)
                    if distance < 40:  # If distance is less than 40 pixels
                        score += 1
                        ball_pos = None

    # Display score and lives
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lives: {lives}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        cv2.putText(frame, "Game Over!", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, "Press 'q' to restart", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Hand Ball Game', frame)

    # Check for 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if game_over:
            reset_game()
        else:
            break

cap.release()
cv2.destroyAllWindows()