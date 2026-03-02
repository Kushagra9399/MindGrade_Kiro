import os
import json
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

from groq import Groq

from schemas import Question, UserResponse, MarkingScheme, QuizResult

# ==========================
# Setup
# ==========================

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-120b"  # You can change to gpt-oss-20b if needed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# REQUEST SCHEMAS
# ==========================


class GenerateQuizRequest(BaseModel):
    topic: str
    difficulty: str
    count: int = 5
    classLevel: str = "10"


class EvaluateQuizRequest(BaseModel):
    questions: List[Question]
    userResponses: List[UserResponse]
    markingScheme: MarkingScheme


# ==========================
# Helper: Call Groq Model
# ==========================


def call_groq(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You must return strictly valid JSON. No markdown. No explanation outside JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    # Clean accidental markdown wrapping
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]

    return json.loads(content)


# ==========================
# GENERATE QUIZ
# ==========================


@app.post("/generate-quiz", response_model=List[Question])
async def generate_quiz(request: GenerateQuizRequest):
    prompt = f"""
Generate {request.count} multiple-choice questions on the topic(s) of: "{request.topic}".
Target Audience: Student in Class/Grade {request.classLevel}.
Difficulty Level: "{request.difficulty}".

Rules:
- Competitive exam level.
- Multi-step logical reasoning required.
- Use '$' for inline math.
- Double escape backslashes.
- Output strictly valid JSON array.

Required structure:
[
  {{
    "id": "string",
    "text": "string",
    "options": [
      {{ "id": "string", "text": "string" }}
    ],
    "correctOptionId": "string",
    "correctReasoning": "string"
  }}
]
"""

    try:
        parsed = call_groq(prompt)

        return [Question(**q) for q in parsed]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# EVALUATE QUIZ
# ==========================


@app.post("/evaluate-quiz", response_model=QuizResult)
async def evaluate_quiz(request: EvaluateQuizRequest):
    data_for_ai = []

    for q in request.questions:
        user_resp = next(
            (ur for ur in request.userResponses if ur.questionId == q.id), None
        )

        data_for_ai.append(
            {
                "questionId": q.id,
                "questionText": q.text,
                "options": [opt.dict() for opt in q.options],
                "correctOptionId": q.correctOptionId,
                "goldenReasoning": q.correctReasoning,
                "userSelectedOptionId": user_resp.selectedOptionId
                if user_resp
                else "NO_ANSWER",
                "userReasoning": user_resp.reasoning if user_resp else "NO_REASONING",
            }
        )

    prompt = f"""
Evaluate the following student responses.

SCORING RULES:
- Correct: +{request.markingScheme.answerPoints}
- Incorrect: -{request.markingScheme.negativeMarking}
- Not Attempted: 0
- Reason Score max: {request.markingScheme.reasonPoints}

Return strictly valid JSON with this structure:

{{
  "evaluations": [
    {{
      "questionId": "string",
      "answerScore": number,
      "reasonScore": number,
      "totalScore": number,
      "isAnswerCorrect": boolean,
      "reasonFeedback": "string",
      "correctReasoning": "string",
      "correctOptionId": "string"
    }}
  ],
  "totalScore": number,
  "maxScore": number,
  "summary": "string"
}}

Data:
{json.dumps(data_for_ai, indent=2)}
"""

    try:
        parsed = call_groq(prompt)
        return QuizResult(**parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
