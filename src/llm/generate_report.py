import requests


def generate_report(
    prediction_text,
    risk_score,
    alert_level,
    top_features,
    model_name="phi3"
):

    feature_text = ""

    for feature, contribution in top_features:

        direction = (
            "Increased Mule Risk"
            if contribution > 0
            else "Decreased Mule Risk"
        )

        feature_text += (
            f"{feature}: "
            f"{contribution:.4f} "
            f"({direction})\n"
        )

    prompt = f"""
You are an expert banking fraud analyst.

IMPORTANT RULES:

- The dataset is anonymized.
- Features such as F3912, F3484, F2686, etc. have unknown business meanings.
- Do NOT assume what these features represent.
- Do NOT mention transactions, money laundering, transfers, deposits, withdrawals, or any real-world banking activity unless explicitly provided.
- Refer to them only as model features.
- Explain only how they contributed to the prediction.

Prediction: {prediction_text}

Risk Score: {risk_score:.2f}%

Alert Level: {alert_level}

Top Feature Contributions:

{feature_text}

Explain:

1. Why the model classified the account this way.
2. Why the risk score is high or low.
3. Which model features contributed most.
4. Whether further investigation is recommended.

Use simple professional language.
Maximum 150 words.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        return response.json()["response"]

    except Exception as e:

        return f"LLM Report Generation Failed: {e}"