import gradio as gr
import csv
from model import predict_ticket, queue_encoder, priority_encoder

session_feedback = []


def ui_predict(subject, body):

    result = predict_ticket(subject, body)

    warning = ""
    if result["uncertain"]:
        warning = "⚠️ **Model is uncertain. Please verify prediction.**"

    return (
        result["department"],
        result["department_confidence"],
        result["priority"],
        result["priority_confidence"],
        result["highlighted_text"],
        result["important_words"],
        result["queue_reason"],
        result["competing_department"],
        result["confidence_gap"],
        warning,
        result["queue_distribution"],
        result["similar_examples"],
        result["competing_words"]
    )


def save_feedback(subject, body, dept, prio):
    session_feedback.append((subject, body, dept, prio))
    return "Correction stored for this session."


def commit_feedback():
    if not session_feedback:
        return "No feedback to commit."

    with open("feedback_log.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(session_feedback)

    session_feedback.clear()
    return "Feedback saved successfully."


with gr.Blocks() as demo:

    gr.Markdown("# 📨 Customer Support Ticket Routing System")

    subject = gr.Textbox(label="Subject")
    body = gr.Textbox(label="Body", lines=6)

    submit_btn = gr.Button("Classify Ticket")

    gr.Markdown("## 🔎 Prediction Results")

    dept_output = gr.Textbox(label="Department")
    dept_conf_output = gr.Number(label="Department Confidence")

    prio_output = gr.Textbox(label="Priority")
    prio_conf_output = gr.Number(label="Priority Confidence")

    gr.Markdown("## 🧠 Explanation")

    highlight_output = gr.HTML()
    important_output = gr.JSON(label="Top Influential Words")
    reason_output = gr.Textbox(label="Confidence Explanation")
    competing_output = gr.Textbox(label="Competing Department")
    gap_output = gr.Number(label="Confidence Gap")
    warning_output = gr.Markdown()
    distribution_output = gr.JSON(label="Department Probability Distribution")
    similar_output = gr.JSON(label="Most Similar Historical Tickets")
    competing_words_output = gr.JSON(label="Words Supporting Competing Department")

    gr.Markdown("## ❗ Correct The Prediction")

    correct_dept = gr.Dropdown(
        choices=list(queue_encoder.classes_),
        label="Correct Department"
    )

    correct_prio = gr.Dropdown(
        choices=list(priority_encoder.classes_),
        label="Correct Priority"
    )

    feedback_btn = gr.Button("Submit Correction")
    commit_btn = gr.Button("Contribute Session Feedback")

    feedback_status = gr.Textbox()
    commit_status = gr.Textbox()

    submit_btn.click(
        ui_predict,
        inputs=[subject, body],
        outputs=[
            dept_output,
            dept_conf_output,
            prio_output,
            prio_conf_output,
            highlight_output,
            important_output,
            reason_output,
            competing_output,
            gap_output,
            warning_output,
            distribution_output,
            similar_output,
            competing_words_output
        ]
    )

    feedback_btn.click(
        save_feedback,
        inputs=[subject, body, correct_dept, correct_prio],
        outputs=feedback_status
    )

    commit_btn.click(
        commit_feedback,
        outputs=commit_status
    )

demo.launch()