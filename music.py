import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import psycopg2
import os

# Load your ML model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
st.title("Emotional based music")
 
# Initialize mediapipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Connect to PostgreSQL database
def connect_to_db():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="12345",
            host="localhost",
            port="5432",
            database="postgres"
        )
        cursor = connection.cursor()
        return connection, cursor
    except (Exception) as error:
        st.error("Error connecting to database")
        return None, None

# Function to create the "Emotion" table in the database
def create_emotion_table():
    try:
        connection, cursor = connect_to_db()
        if connection and cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Emotion (
                    recommendation_id SERIAL PRIMARY KEY,
                    language VARCHAR(40),
                    artist VARCHAR(50),
                    emotion VARCHAR(20)
                )
            ''')
            connection.commit()
            print("Emotion table created successfully")
        else:
            st.error("Error connecting to database")
    except Exception as e:
        st.error(f"Error creating Emotion table: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to create the "User" table in the database
def create_user_table():
    try:
        connection, cursor = connect_to_db()
        if connection and cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS "user_details" (
                    user_id SERIAL PRIMARY KEY,
                    username VARCHAR(50),
                    email VARCHAR(50), 
                    preferred_language VARCHAR(20),
                    favorite_artists TEXT[],
                    preferred_genres TEXT[]
                )
            ''')
            connection.commit()
            print("User table created successfully")
        else:
            st.error("Error connecting to database")
    except Exception as e:
        st.error(f"Error creating User table: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to insert input values into the "Emotion" table
def insert_input_values(lang, artist , emotion):
    connection, cursor = connect_to_db()
    if connection and cursor:
        try:
            cursor.execute("INSERT INTO Emotion (language, artist, emotion) VALUES (%s, %s, %s)", (lang, artist, emotion))
            connection.commit()
            st.success("Input values inserted into the Emotion table successfully")
        except (Exception) as error:
            st.error(f"Error inserting input values into the Emotion table: {error}")
        finally:
            if connection:
                cursor.close()
                connection.close()

# Function to insert user details into the "User" table
def insert_user_details(username, email, preferred_language, favorite_artists, preferred_genres):
    connection, cursor = connect_to_db()
    if connection and cursor:
        try:
            cursor.execute("""
                INSERT INTO "user_details" (username, email, preferred_language, favorite_artists, preferred_genres)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, email, preferred_language, favorite_artists, preferred_genres))
            connection.commit()
            st.success("User details inserted into the User table successfully")
        except (Exception) as error:
            st.error(f"Error inserting user details: {error}")
        finally:
            if connection:
                cursor.close()
                connection.close()

# Function to retrieve input values from the "Emotion" table
def retrieve_input_values():
    connection, cursor = connect_to_db()
    input_values = None
    if connection and cursor:
        try:
            cursor.execute("SELECT language, artist, emotion FROM Emotion ORDER BY id DESC LIMIT 1")
            input_values = cursor.fetchone()
        except (Exception,) as error:
            st.error(f"Error retrieving input values from the Emotion table: {error}")
        finally:
            if connection:
                cursor.close()
                connection.close()
    return input_values

# Emotion Processor class for video processing
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if len(lst) > 0:
                lst = np.array(lst).reshape(1, -1)
                pred = label[np.argmax(model.predict(lst))]
                print(pred)
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            else:
                st.warning("No landmarks detected. Please ensure your face and hands are visible.")
            
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Check if the "Emotion" table exists, and if not, create it
create_emotion_table()

# Check if the "User" table exists, and if not, create it
create_user_table()

# Get user details from the user
st.sidebar.title("User Details")
username = st.sidebar.text_input("Username")
email = st.sidebar.text_input("Email")
preferred_language = st.sidebar.text_input("Preferred Language")
favorite_artists = st.sidebar.text_area("Favorite Artists (comma-separated)")
preferred_genres = st.sidebar.text_area("Preferred Genres (comma-separated)")

# Convert favorite_artists and preferred_genres to lists
favorite_artists_list = [artist.strip() for artist in favorite_artists.split(",") if artist.strip()]
preferred_genres_list = [genre.strip() for genre in preferred_genres.split(",") if genre.strip()]

# Insert user details into the database when the button is clicked
if st.sidebar.button("Save User Details"):
    if username and email:
        insert_user_details(username, email, preferred_language, favorite_artists_list, preferred_genres_list)
    else:
        st.sidebar.warning("Please enter username and email")

# Get language and artist inputs
lang = st.text_input("Language")
artist = st.text_input("Artist")

# Display video stream and emotion detection
if lang and artist:
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button to recommend songs
btn = st.button("Recommend me songs")
if btn:
    emotion = np.load("emotion.npy")[0] if "emotion.npy" in os.listdir() else ""
    if not emotion:
        st.warning("Please let me capture your emotion first")
    else:
        insert_input_values(lang, artist, emotion)
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{artist}")
        np.save("emotion.npy", np.array([""]))
