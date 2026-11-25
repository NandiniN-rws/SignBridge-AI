
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy.orm import Session
from sqlalchemy import desc

from .core.db import Base, engine, get_db
from .core.config import settings

from . import models, schemas

from .agents.genai import (
    patient_message,
    simplify,
    doctor_response,
    emergency_flags
)

from .services.sign_model import (
    Predictor,
    NUM_FRAMES,
    IMAGE_SIZE
)

from .services.medical_image_model import (
    MedicalImagePredictor
)


# ==========================================
# VIDEO REQUEST SCHEMA
# ==========================================

class VideoFramesIn(BaseModel):
    frames: list


# ==========================================
# SIGN DICTIONARY
# ==========================================

SIGN_DICTIONARY = [

    {
        "sign": "accident",
        "category": "Emergency",
        "meaning": "An accident or injury event.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "call",
        "category": "Emergency",
        "meaning": "Request to call someone.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "doctor",
        "category": "Healthcare",
        "meaning": "Doctor or medical professional.",
        "dataset": "Both datasets"
    },

    {
        "sign": "help",
        "category": "Emergency",
        "meaning": "Request for help or assistance.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "hot",
        "category": "Symptoms",
        "meaning": "Feeling hot or indicating heat.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "lose",
        "category": "General",
        "meaning": "Lose or losing something; confirm intended context.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "pain",
        "category": "Symptoms",
        "meaning": "Pain or physical discomfort.",
        "dataset": "Both datasets"
    },

    {
        "sign": "thief",
        "category": "Emergency",
        "meaning": "A thief or theft-related situation.",
        "dataset": "Emergency ISL video dataset"
    },

    {
        "sign": "afraid",
        "category": "Feelings",
        "meaning": "Feeling afraid or frightened.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "agree",
        "category": "General",
        "meaning": "To agree or indicate agreement.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "assistance",
        "category": "Emergency",
        "meaning": "Requesting assistance.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "bad",
        "category": "General",
        "meaning": "Bad or not good.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "become",
        "category": "General",
        "meaning": "Become or change into a state.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "college",
        "category": "General",
        "meaning": "College or educational institution.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "from",
        "category": "General",
        "meaning": "Indicating origin or starting point.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "pray",
        "category": "General",
        "meaning": "Prayer or praying.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "secondary",
        "category": "General",
        "meaning": "Secondary or second-level.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "skin",
        "category": "Body",
        "meaning": "Skin or the body's outer covering.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "small",
        "category": "General",
        "meaning": "Small in size.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "specific",
        "category": "General",
        "meaning": "Specific or particular.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "stand",
        "category": "Actions",
        "meaning": "Stand or standing.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "today",
        "category": "Time",
        "meaning": "Today or this day.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "warn",
        "category": "Emergency",
        "meaning": "To warn or alert someone.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "which",
        "category": "Questions",
        "meaning": "Which one; asking to identify a choice.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "work",
        "category": "General",
        "meaning": "Work or working.",
        "dataset": "Medical ISL image dataset"
    },

    {
        "sign": "you",
        "category": "People",
        "meaning": "You; referring to the person being addressed.",
        "dataset": "Medical ISL image dataset"
    }

]


# ==========================================
# DATABASE
# ==========================================

Base.metadata.create_all(bind=engine)


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(

    title="SignBridge AI",

    version="2.0.0",

    description=(
        "Assistive communication prototype; "
        "not a certified medical interpreter."
    )

)


# ==========================================
# CORS
# ==========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        x.strip()
        for x in settings.cors_origins.split(",")
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ==========================================
# LOAD MODELS
# ==========================================

video_predictor = Predictor()

medical_predictor = MedicalImagePredictor()


# ==========================================
# BASIC ROUTES
# ==========================================

@app.get("/")
def root():

    return {
        "name": "SignBridge AI",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ==========================================
# MODEL STATUS
# ==========================================

@app.get("/api/model/status")
def model_status():

    return {

        "video": {

            "ready": video_predictor.model_ready,

            "device": str(video_predictor.device),

            "expected_shape": [
                NUM_FRAMES,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            ],

            "labels": video_predictor.labels,

            "architecture": "CNN + BiLSTM + Transformer"

        },

        "medical_image": {

            "ready": medical_predictor.model is not None,

            "device": medical_predictor.device,

            "labels": medical_predictor.labels,

            "message": (
                medical_predictor.error
                or "Checkpoint loaded"
            )

        }

    }


# ==========================================
# VIDEO SIGN PREDICTION - V2
# ==========================================

@app.post("/api/sign/predict")
def predict(payload: VideoFramesIn):

    if len(payload.frames) != NUM_FRAMES:

        raise HTTPException(
            status_code=422,
            detail=f"Expected {NUM_FRAMES} RGB video frames."
        )

    for frame in payload.frames:

        if (
            not isinstance(frame, list)
            or len(frame) != IMAGE_SIZE
        ):

            raise HTTPException(
                status_code=422,
                detail=f"Each frame must have {IMAGE_SIZE} rows."
            )

        for row in frame:

            if (
                not isinstance(row, list)
                or len(row) != IMAGE_SIZE
            ):

                raise HTTPException(
                    status_code=422,
                    detail=f"Each frame must be {IMAGE_SIZE}x{IMAGE_SIZE}."
                )

            for pixel in row:

                if (
                    not isinstance(pixel, list)
                    or len(pixel) != 3
                    or any(
                        not isinstance(value, (int, float))
                        for value in pixel
                    )
                ):

                    raise HTTPException(
                        status_code=422,
                        detail="Each pixel must contain three RGB values."
                    )

    return video_predictor.predict(
        payload.frames
    )


# ==========================================
# MEDICAL IMAGE PREDICTION
# ==========================================

@app.post("/api/medical-image/predict")
def predict_medical_image(
    payload: schemas.MedicalImageIn
):

    return medical_predictor.predict_base64(
        payload.image_base64
    )


# ==========================================
# GENAI PATIENT MESSAGE
# ==========================================

@app.post("/api/ai/patient-message")
async def improve(
    payload: schemas.TextIn
):

    return await patient_message(
        payload.text,
        payload.context
    )


# ==========================================
# SIMPLIFY MEDICAL LANGUAGE
# ==========================================

@app.post("/api/ai/simplify")
async def simplify_route(
    payload: schemas.TextIn
):

    return await simplify(
        payload.text
    )


# ==========================================
# DOCTOR RESPONSE
# ==========================================

@app.post("/api/ai/doctor-response")
async def doctor_response_route(
    payload: schemas.DoctorResponseIn
):

    return await doctor_response(
        payload.text,
        payload.language
    )


# ==========================================
# LEARN ISL SIGNS
# ==========================================

@app.get("/api/learn/signs")
def learn_signs():

    return SIGN_DICTIONARY


# ==========================================
# EMERGENCY KEYWORD CHECK
# ==========================================

@app.post("/api/emergency/check")
def emergency(
    payload: schemas.TextIn
):

    flags = emergency_flags(
        payload.text
    )

    return {

        "flags": flags,

        "attention_required": bool(flags),

        "disclaimer": (
            "Keyword alert only; not clinical triage."
        )

    }


# ==========================================
# CREATE CONVERSATION
# ==========================================

@app.post("/api/conversations")
def create(
    payload: schemas.ConversationIn,
    db: Session = Depends(get_db)
):

    conversation = models.Conversation(
        title=payload.title
    )

    db.add(conversation)

    db.commit()

    db.refresh(conversation)

    return {

        "id": conversation.id,

        "title": conversation.title

    }


# ==========================================
# LIST CONVERSATIONS
# ==========================================

@app.get("/api/conversations")
def list_conversations(
    db: Session = Depends(get_db)
):

    rows = (

        db.query(models.Conversation)

        .order_by(
            desc(models.Conversation.created_at)
        )

        .all()

    )

    return [

        {
            "id": c.id,
            "title": c.title
        }

        for c in rows

    ]


# ==========================================
# GET CONVERSATION
# ==========================================

@app.get("/api/conversations/{cid}")
def get_conversation(
    cid: int,
    db: Session = Depends(get_db)
):

    conversation = (

        db.query(models.Conversation)

        .filter_by(id=cid)

        .first()

    )

    if not conversation:

        raise HTTPException(
            404,
            "Conversation not found"
        )

    return {

        "id": conversation.id,

        "title": conversation.title,

        "messages": [

            {

                "id": m.id,

                "role": m.role,

                "content": m.content,

                "confirmed": m.confirmed,

                "source": m.source

            }

            for m in conversation.messages

        ]

    }


# ==========================================
# ADD MESSAGE
# ==========================================

@app.post("/api/conversations/{cid}/messages")
def add_message(
    cid: int,
    payload: schemas.MessageIn,
    db: Session = Depends(get_db)
):

    conversation = (

        db.query(models.Conversation)

        .filter_by(id=cid)

        .first()

    )

    if not conversation:

        raise HTTPException(
            404,
            "Conversation not found"
        )

    message = models.Message(

        conversation_id=cid,

        **payload.model_dump()

    )

    db.add(message)

    db.commit()

    db.refresh(message)

    flags = (

        emergency_flags(message.content)

        if message.role == "patient"

        else []

    )

    return {

        "id": message.id,

        "role": message.role,

        "content": message.content,

        "confirmed": message.confirmed,

        "source": message.source,

        "emergency_flags": flags

    }


# ==========================================
# CONVERSATION REPORT
# ==========================================

@app.get("/api/reports/{cid}/summary")
def report(
    cid: int,
    db: Session = Depends(get_db)
):

    conversation = (

        db.query(models.Conversation)

        .filter_by(id=cid)

        .first()

    )

    if not conversation:

        raise HTTPException(
            404,
            "Conversation not found"
        )

    return {

        "conversation_id": cid,

        "title": conversation.title,

        "entries": [

            {

                "role": m.role,

                "message": m.content,

                "confirmed": m.confirmed

            }

            for m in conversation.messages

        ],

        "note": (
            "Communication summary only; "
            "not a diagnosis or certified medical record."
        )

    }
