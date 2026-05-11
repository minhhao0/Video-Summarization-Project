# Video-Summarization-Project
## Installation Guide
### 1. Clone git repository
`git clone https://github.com/minhhao0/Video-Summarization-Project.git`
### 2. Create directory
```
cd Video-Summarization-Project
mkdir audio
mkdir audio_chunk
mkdir model
mkdir text
```
### 3. Download model weight
Download model weight and config from [GG Drive](https://drive.google.com/file/d/1u6h41io8EuOuhgCvwSZOaANbPIQU2-2t/view?usp=sharing)
and then unzip folder model then copy the 'model' folder to project
### 4. Install ffmpeg
Install ffmeg from [FFmeg](https://ffmpeg.org/download.html).
In Window can use this command then restart your program
`winget install ffmpeg`
### 5. Install requirements library
`pip install -r requirements.txt`
### 6. Install nodejs.
Down load Node js from [Node Js](https://nodejs.org/en/download)
then 
```
cd frontend
npm install
npm start
```
### 7. Start FAST Api backend
```
cd backend/api
uvicorn main:app --reload
```
## RUNNING ON GG COLAB
### Run deploy_server.ipynb on gg colab. 
### 1. Go to [Ngrok Site](https://ngrok.com/) to get your authentication token
### 2. Paste your token in 
```
NGROK_AUTH_TOKEN = 'your token'
```
### 3. Run notebook and start front-end
### NOTICE: Your must enable your GPU on GG Colab to increase speed.





