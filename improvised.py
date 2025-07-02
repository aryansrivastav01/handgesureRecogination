#this is a improvised version of hand gesture recoginition with more hand gesture coordinate and finger placement 
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the hand gesture recognition model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to determine finger states (1 = extended, 0 = folded)
def get_finger_states(hand_landmarks):
    finger_states = []

    # Thumb (compare x-coordinates for left/right hand)
    if hand_landmarks[4].x < hand_landmarks[3].x:
        finger_states.append(1)  # Thumb extended
    else:
        finger_states.append(0)

    # Other fingers (compare y-coordinates)
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
            finger_states.append(1)  # Finger extended
        else:
            finger_states.append(0)

    return finger_states

# Function to recognize gestures based on finger states
def recognize_gesture(finger_states):
    gestures = {
        (0, 0, 0, 0, 0): "Fist",
        (1, 1, 1, 1, 1): "Hello",
        (1, 0, 0, 0, 0): "Thumbs Up",
        (0, 1, 1, 0, 0): "Victory",
        (0, 1, 1, 1, 1): "Nice",
        (0, 1, 0, 0, 0): "Pointing",
        (0, 1, 0, 1, 0): "Rock",
        (1, 1, 0, 0, 1): "Spider-Man"
    }

    return gestures.get(tuple(finger_states), "Gesture not recognized")

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand annotations and recognize gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_states = get_finger_states(hand_landmarks.landmark)
            gesture = recognize_gesture(finger_states)

            # Display gesture name
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
