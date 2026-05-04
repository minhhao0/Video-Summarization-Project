from fastapi import FastAPI, File, UploadFile
from audio_to_text_api.model import Mp4_to_text
from summary_api.model import Summarizer
from topic_model_api.model import TopicModel
from fastapi.middleware.cors import  CORSMiddleware
origins = [
    "http://localhost:3000",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
audio_to_text_model=Mp4_to_text('../../model/Phowisper')
summary_model=Summarizer('../../model/mt5')
topic_model=TopicModel('../../model/vietnamese-embedding')
@app.post("/api/summary-file")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"../../audio/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    print('Processing audio to text')
    text=audio_to_text_model.op_vi(file_location)
    print("Processing topic modeling")
    topic=topic_model.get_topic(paragraph=text)
    print("Processing text summary")
    summary=summary_model.run('../../text/output.txt')
    return {'topic':topic,
            'summary':summary}
@app.post("/api/summary-link")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"../../audio/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    text=audio_to_text_model.op_vi(f'../{file_location}')
    topic=topic_model.get_topic(paragraph=text)
    summary=summary_model.run('../../text/output.txt')
    return {'msg':'save file audio successfully.'}

