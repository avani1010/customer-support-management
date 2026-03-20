import gradio as gr
from model import predict_ticket


def ui_predict(subject, body):
    result = predict_ticket(subject, body)

    return (
        result["department"],
        result["department_confidence"],
        result["priority"],
        result["priority_confidence"]
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📨 Customer Support Ticket Routing System")
    gr.Markdown("Raise a customer complaint.")

    with gr.Row():
        subject = gr.Textbox(
            label="Subject",
            placeholder="Enter ticket subject..."
        )

    body = gr.Textbox(
        label="Body",
        lines=6,
        placeholder="Describe the issue in detail..."
    )

    submit_btn = gr.Button("Classify Ticket")

    gr.Markdown("## 🔎 Prediction Results")

    with gr.Row():
        dept_output = gr.Textbox(label="Department")
        dept_conf_output = gr.Number(label="Department Confidence")

    with gr.Row():
        prio_output = gr.Textbox(label="Priority")
        prio_conf_output = gr.Number(label="Priority Confidence")

    submit_btn.click(
        ui_predict,
        inputs=[subject, body],
        outputs=[
            dept_output,
            dept_conf_output,
            prio_output,
            prio_conf_output
        ]
    )

demo.launch()
