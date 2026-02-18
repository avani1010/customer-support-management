def predict_ticket(subject: str, body: str) -> dict:
    return {
        "department": "Billing",
        "department_confidence": 0.87,
        "priority": "High",
        "priority_confidence": 0.76
    }