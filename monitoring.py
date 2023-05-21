import streamlit as st
import cv2
import main


if __name__ == '__main__':
    opencamera = st.button("Open Camera")
    if opencamera:
        main.bicepsCurls_left(0)
