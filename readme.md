# Pose Feedback

Pose Feedback is a Python app that uses computer vision and natural language generation to provide personalized and relative feedback for people based on their pose metrics.

## Features

- Detects people in real-time using OpenCV library
- Estimates holistic landmarks of each person using MediaPipe Holistic model
- Calculates pose metrics such as angles, distances, symmetries, and balances for each person using NumPy or SciPy library
- Generates feedback messages for each person based on their pose metrics using natural language generation (NLG) techniques
- Compares pose metrics of different people and generates relative feedback using NumPy or SciPy library
- Displays feedback messages and comparison results on the frame using OpenCV library

## Installation

To install Pose Feedback, you need to have Python 3.6 or higher and pip installed on your system. Then, you can clone this repository and install the required dependencies using the following commands:

```bash
git clone https://github.com/pose-feedback/pose-feedback.git
cd pose-feedback
pip install -r requirements.txt
