import gradio as gr

if __name__ == "__main__":
    gr.themes.builder()
import gradio as gr

theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="teal",
    neutral_hue=gr.themes.Color(neutral_100="#f3f4f6", neutral_200="#e5e7eb", neutral_300="#d1d5db", neutral_400="#9ca3af", neutral_50="#f9fafb", neutral_500="#6b7280", neutral_600="#4b5563", neutral_700="#374151", neutral_800="#1f2937", neutral_900="#47a9c2", neutral_950="#0b0f19"),
)

