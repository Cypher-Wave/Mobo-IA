import io
import glob
import base64
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
IMG_HEIGHT, IMG_WIDTH = 224, 224
LAST_CONV_LAYER_NAME = "out_relu"

MODEL_PATH = BASE_DIR / "ia_lichia" / "modelo_lichia.keras"
LOGO_PATH = BASE_DIR / "LOGO SPLASH SCREEN.png"
CSS_PATH = BASE_DIR / "styles.css"
MAX_DISPLAY_CONFIDENCE = 0.982


# -----------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Mobo - Análise de Lichias",
    page_icon="🍒",
    layout="wide"
)


# -----------------------------
# ESTILOS
# -----------------------------
def load_css():
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def image_to_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return ""


def pil_image_to_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# -----------------------------
# ESTADO
# -----------------------------
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None

    if "selected_analysis_index" not in st.session_state:
        st.session_state.selected_analysis_index = 0

    if "last_history_key" not in st.session_state:
        st.session_state.last_history_key = None


# -----------------------------
# CARREGAMENTO DE RECURSOS
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def get_example_images():
    madura = sorted(
        glob.glob(str(BASE_DIR / "ia_lichia" / "dataset" / "test" / "madura" / "*"))
    )
    nao_madura = sorted(
        glob.glob(str(BASE_DIR / "ia_lichia" / "dataset" / "test" / "nao_madura" / "*"))
    )

    examples = {}
    if madura:
        examples["Exemplo madura"] = madura[0]
    if nao_madura:
        examples["Exemplo não madura"] = nao_madura[0]

    return examples


# -----------------------------
# PROCESSAMENTO DE IMAGEM
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_resized, image_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    base_model = model.get_layer("mobilenetv2_1.00_224")

    # Cria modelo funcional passando pelo base_model e depois pelas camadas restantes
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    conv_outputs = base_model(inputs)
    
    # Pega saída da última conv dentro do base_model
    inner_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(last_conv_layer_name).output
    )

    with tf.GradientTape() as tape:
        conv_out = inner_model(img_array)
        tape.watch(conv_out)
        
        # Passa conv_out pelo resto do modelo manualmente
        x = conv_out
        for layer in model.layers[1:]:  # pula o base_model
            x = layer(x)
        predictions = x
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize(original_image.size)

    original_array = np.array(original_image, dtype=np.float32)
    heatmap_array = np.array(jet_heatmap, dtype=np.float32)

    superimposed = heatmap_array * alpha + original_array
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


# -----------------------------
# REGRAS DE APRESENTAÇÃO
# -----------------------------
def limit_display_confidence(confidence):
    return min(float(confidence), MAX_DISPLAY_CONFIDENCE)


def get_result_style(confidence):
    if confidence >= 0.90:
        return {
            "bg": "#F7F1E8",
            "border": "#30391F",
            "title": "Alta confiança",
            "text": "O modelo identificou o resultado com alta segurança."
        }
    if confidence >= 0.70:
        return {
            "bg": "#FFF7E8",
            "border": "#B58A4A",
            "title": "Confiança moderada",
            "text": "O modelo encontrou um resultado plausível, mas com segurança intermediária."
        }
    return {
        "bg": "#FFF0F4",
        "border": "#C00646",
        "title": "Baixa confiança",
        "text": "O modelo encontrou um resultado incerto. Vale a pena testar outra imagem."
    }


def get_prediction_text(predicted_class, confidence):
    display_confidence = limit_display_confidence(confidence)
    if predicted_class == "madura":
        return f"A IA identificou que a lichia está madura com confiança de {display_confidence:.2%}."
    return f"A IA identificou que a lichia está não madura com confiança de {display_confidence:.2%}."


# -----------------------------
# EXPORTAÇÃO
# -----------------------------
def safe_font(size=28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_report_image(original_image, heatmap_image, predicted_class, confidence, timestamp_text):
    display_confidence = limit_display_confidence(confidence)
    width = 1600
    height = 980
    bg = (239, 230, 216)
    accent = (192, 6, 70)
    green = (48, 57, 31)
    white = (255, 252, 247)
    text = (48, 57, 31)
    soft = (115, 94, 87)

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    title_font = safe_font(54)
    subtitle_font = safe_font(28)
    text_font = safe_font(34)
    small_font = safe_font(24)

    draw.rounded_rectangle((40, 35, 1560, 170), radius=30, fill=accent)
    draw.text((70, 60), "Relatório de análise - Mobo IA Lichia", fill=white, font=title_font)
    draw.text((70, 122), f"Gerado em: {timestamp_text}", fill=white, font=subtitle_font)

    result_color = green if confidence >= 0.90 else accent
    draw.rounded_rectangle((40, 205, 1560, 355), radius=26, fill=white, outline=accent, width=6)
    draw.text((70, 235), f"Classe prevista: {predicted_class}", fill=text, font=text_font)
    draw.text((70, 285), f"Confiança: {display_confidence:.2%}", fill=result_color, font=text_font)

    img_w, img_h = 650, 520
    original_resized = original_image.resize((img_w, img_h))
    heatmap_resized = heatmap_image.resize((img_w, img_h))

    canvas.paste(original_resized, (70, 400))
    canvas.paste(heatmap_resized, (880, 400))

    draw.rounded_rectangle((60, 390, 730, 930), radius=18, outline=accent, width=6)
    draw.rounded_rectangle((870, 390, 1540, 930), radius=18, outline=accent, width=6)

    draw.text((280, 935), "Imagem original", fill=soft, font=small_font)
    draw.text((1060, 935), "Heatmap da atenção da IA", fill=soft, font=small_font)

    return canvas


def pil_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def pil_to_pdf_bytes(image):
    buffer = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffer, format="PDF")
    buffer.seek(0)
    return buffer


# -----------------------------
# HISTÓRICO
# -----------------------------
def add_to_history(file_name, predicted_class, confidence):
    display_confidence = limit_display_confidence(confidence)
    entry = {
        "file_name": file_name,
        "predicted_class": predicted_class,
        "confidence": f"{display_confidence:.2%}",
        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


def register_history_once(result):
    history_key = (
        result["file_name"],
        result["predicted_class"],
        f"{result['confidence']:.4f}"
    )

    if st.session_state.last_history_key != history_key:
        add_to_history(
            result["file_name"],
            result["predicted_class"],
            result["confidence"]
        )
        st.session_state.last_history_key = history_key


# -----------------------------
# ANÁLISE
# -----------------------------
def analyze_image(image, image_source_name, model):
    processed_image, image_array = preprocess_image(image)

    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_class = "não madura"
        confidence = prediction
    else:
        predicted_class = "madura"
        confidence = 1 - prediction

    heatmap = make_gradcam_heatmap(image_array, model, LAST_CONV_LAYER_NAME)
    heatmap_image = overlay_heatmap_on_image(processed_image, heatmap)

    return {
        "file_name": image_source_name,
        "processed_image": processed_image,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "display_confidence": limit_display_confidence(confidence),
        "heatmap_image": heatmap_image,
        "style": get_result_style(confidence),
        "prediction_text": get_prediction_text(predicted_class, confidence)
    }


def build_selected_images(uploaded_files):
    selected_images = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            selected_images.append({
                "name": uploaded_file.name,
                "image": Image.open(uploaded_file).convert("RGB")
            })
    elif st.session_state.selected_example is not None:
        selected_images.append({
            "name": Path(st.session_state.selected_example).name,
            "image": Image.open(st.session_state.selected_example).convert("RGB")
        })

    if not selected_images:
        st.session_state.selected_analysis_index = 0
    elif st.session_state.selected_analysis_index >= len(selected_images):
        st.session_state.selected_analysis_index = 0

    return selected_images


# -----------------------------
# COMPONENTES DE INTERFACE
# -----------------------------
def render_mobo_shell():
    logo_base64 = image_to_base64(LOGO_PATH)
    logo_html = (
        f'<img src="data:image/png;base64,{logo_base64}" class="mobo-logo" alt="MOBO">'
        if logo_base64
        else '<div class="mobo-logo-text">mobo</div>'
    )
    menu_items = [
        ("⌂", "Home"),
        ("⚙", "Braço Mecânico"),
        ("✦", "Análise IA"),
        ("◔", "Dashboard"),
        ("⌁", "Sensores"),
        ("▤", "Relatórios"),
        ("●", "Alertas"),
        ("☼", "Previsão Colheita"),
        ("◉", "Perfil"),
        ("⊕", "Terreno"),
        ("↳", "Logout"),
    ]
    menu_html = "".join(
        f'<div class="mobo-nav-item"><span>{icon}</span><strong>{label}</strong></div>'
        for icon, label in menu_items
    )
    st.markdown(
        f"""
        <aside class="mobo-sidebar">
            <div class="mobo-logo-wrap">{logo_html}</div>
            <nav class="mobo-nav">{menu_html}</nav>
        </aside>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <header class="mobo-header">
            <div>
                <div class="eyebrow">MOBO IA</div>
                <h1>Identificação de Maturação</h1>
            </div>
            <div class="top-actions" aria-label="Ações rápidas">
                <div class="theme-toggle" aria-label="Alternar tema">
                    <span class="theme-symbol">
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                            <circle cx="12" cy="12" r="3.7"></circle>
                            <path d="M12 2.6v2M12 19.4v2M5.4 5.4l1.45 1.45M17.15 17.15l1.45 1.45M2.6 12h2M19.4 12h2M5.4 18.6l1.45-1.45M17.15 6.85l1.45-1.45"></path>
                        </svg>
                    </span>
                    <span class="theme-thumb"></span>
                    <span class="theme-symbol dark">
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M20.2 15.6A7.9 7.9 0 0 1 8.4 3.8a6.5 6.5 0 1 0 11.8 11.8Z"></path>
                        </svg>
                    </span>
                </div>
                <div class="top-icon profile-icon" aria-label="Perfil">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                        <circle cx="12" cy="8.2" r="3.3"></circle>
                        <path d="M5.2 20.2c.85-3.65 3.25-5.5 6.8-5.5s5.95 1.85 6.8 5.5"></path>
                    </svg>
                </div>
                <div class="top-icon notification-icon" aria-label="Notificações">
                    <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M18 9.4a6 6 0 0 0-12 0c0 5.4-2.2 6.45-3 8.15h18c-.8-1.7-3-2.75-3-8.15"></path>
                        <path d="M14.3 20.2a2.55 2.55 0 0 1-4.6 0"></path>
                    </svg>
                    <span class="notification-dot"></span>
                </div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_example_buttons(examples):
    st.markdown('<div class="section-title compact">Teste rápido</div>', unsafe_allow_html=True)
    example_cols = st.columns(3)

    with example_cols[0]:
        if "Exemplo madura" in examples and st.button("Usar exemplo madura", width="stretch"):
            st.session_state.selected_example = examples["Exemplo madura"]
            st.session_state.selected_analysis_index = 0

    with example_cols[1]:
        if "Exemplo não madura" in examples and st.button("Usar exemplo não madura", width="stretch"):
            st.session_state.selected_example = examples["Exemplo não madura"]
            st.session_state.selected_analysis_index = 0

    with example_cols[2]:
        if st.button("Remover exemplo selecionado", width="stretch"):
            st.session_state.selected_example = None
            st.session_state.selected_analysis_index = 0


def render_image_card(image, title, caption=None, card_class="visual-card"):
    caption_html = f'<div class="image-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="mobo-card {card_class}">
            <div class="visual-card-title">{title}</div>
            <div class="image-frame">
                <img src="{pil_image_to_data_uri(image)}" alt="{title}">
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_cards(analysis_results):
    st.markdown('<div class="section-title">Comparação das análises</div>', unsafe_allow_html=True)

    num_columns = min(len(analysis_results), 3)
    comparison_cols = st.columns(num_columns)

    for index, result in enumerate(analysis_results):
        col = comparison_cols[index % num_columns]
        with col:
            st.markdown(
                f"""
                <div class="mobo-card comparison-shell">
                    <div class="image-frame compact">
                        <img src="{pil_image_to_data_uri(result['processed_image'])}" alt="{result['file_name']}">
                    </div>
                    <div class="comparison-image-caption">{result['file_name']}</div>
                    <div class="comparison-card" style="border-left: 6px solid {result['style']['border']};">
                        <div class="comparison-title" style="color:{result['style']['border']};">
                            {result['predicted_class']}
                        </div>
                        <div class="comparison-text">
                            <strong>Confiança:</strong> {result['display_confidence']:.2%}<br>
                            {result['prediction_text']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Ver análise detalhada",
                key=f"select_analysis_{index}",
                width="stretch"
            ):
                st.session_state.selected_analysis_index = index
                st.rerun()

            if st.session_state.selected_analysis_index == index:
                st.markdown(
                    '<div class="selected-chip">Análise selecionada</div>',
                    unsafe_allow_html=True
                )


def render_detailed_result(primary_result):
    st.markdown(
        f"""
        <div class="mobo-card result-card" style="background-color:{primary_result['style']['bg']}; border-color:{primary_result['style']['border']};">
            <div class="card-kicker">Resultado da análise</div>
            <div class="result-title" style="color:{primary_result['style']['border']};">{primary_result['style']['title']}</div>
            <div class="result-text">
                <strong>Arquivo:</strong> {primary_result['file_name']}<br>
                <strong>Classe prevista:</strong> {primary_result['predicted_class']}<br>
                <strong>Confiança:</strong> {primary_result['display_confidence']:.2%}<br><br>
                {primary_result['prediction_text']}<br>
                {primary_result['style']['text']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_metric1, col_metric2 = st.columns(2)

    with col_metric1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Classe prevista</div>
                <div class="metric-value">{primary_result['predicted_class']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_metric2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Confiança</div>
                <div class="metric-value">{primary_result['display_confidence']:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(float(primary_result["display_confidence"]))


def render_detailed_visualization(primary_result):
    st.markdown('<div class="section-title">Visualização detalhada</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        render_image_card(
            primary_result["processed_image"],
            "Preview",
            primary_result["file_name"],
            "visual-card",
        )

    with col2:
        render_image_card(
            primary_result["heatmap_image"],
            "Heatmap",
            "Atenção visual da IA",
            "visual-card",
        )

    with st.expander("Como interpretar o heatmap"):
        st.write(
            "O heatmap destaca as regiões da imagem que mais influenciaram a decisão do modelo. "
            "Áreas mais quentes tendem a indicar onde a rede concentrou mais atenção para classificar a lichia."
        )


def render_export_section(primary_result):
    st.markdown('<div class="section-title">Exportação</div>', unsafe_allow_html=True)

    report_image = make_report_image(
        original_image=primary_result["processed_image"],
        heatmap_image=primary_result["heatmap_image"],
        predicted_class=primary_result["predicted_class"],
        confidence=primary_result["confidence"],
        timestamp_text=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )

    png_buffer = pil_to_png_bytes(report_image)
    pdf_buffer = pil_to_pdf_bytes(report_image)

    export_left, export_col1, export_col2, export_right = st.columns([0.26, 0.24, 0.24, 0.26])

    with export_col1:
        st.download_button(
            label="↓ Baixar PNG",
            data=png_buffer,
            file_name="analise_lichia.png",
            mime="image/png",
            width="stretch"
        )

    with export_col2:
        st.download_button(
            label="↓ Baixar PDF",
            data=pdf_buffer,
            file_name="analise_lichia.pdf",
            mime="application/pdf",
            width="stretch"
        )


def render_history_section():
    st.markdown('<div class="section-title">Histórico de previsões</div>', unsafe_allow_html=True)

    if st.session_state.history:
        history_cols = st.columns(min(len(st.session_state.history), 3))
        for index, item in enumerate(st.session_state.history):
            with history_cols[index % len(history_cols)]:
                st.markdown(
                    f"""
                    <div class="history-card">
                        <strong>{item['predicted_class']}</strong>
                        <span>{item['file_name']}</span>
                        <span>Confiança: {item['confidence']}</span>
                        <small>{item['time']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="empty-state">Nenhuma previsão registrada ainda.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="history-clear-spacer"></div>', unsafe_allow_html=True)
    clear_left, clear_action, clear_right = st.columns([0.36, 0.28, 0.36])
    with clear_action:
        if st.button("Limpar histórico", width="stretch"):
            st.session_state.history = []
            st.session_state.last_history_key = None
            st.rerun()


# -----------------------------
# FLUXO PRINCIPAL
# -----------------------------
load_css()
init_session_state()

model = load_model()
examples = get_example_images()

render_mobo_shell()
render_header()

workspace_col, result_col = st.columns([1.04, 0.96], gap="large")

with workspace_col:
    st.markdown(
        """
        <div class="mobo-card upload-intro">
            <div class="card-kicker">Entrada de imagem</div>
            <h2>Envie uma lichia para análise</h2>
            <p>Use uma foto em JPG ou PNG. A IA mantém a análise original e aplica apenas um limite visual na confiança exibida.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Escolha uma ou mais imagens",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    render_example_buttons(examples)

selected_images = build_selected_images(uploaded_files)
analysis_results = []
primary_result = None

if selected_images:
    analysis_results = [
        analyze_image(item["image"], item["name"], model)
        for item in selected_images
    ]
    primary_result = analysis_results[st.session_state.selected_analysis_index]
    register_history_once(primary_result)

with workspace_col:
    if selected_images and primary_result:
        st.markdown('<div class="section-title compact">Preview selecionado</div>', unsafe_allow_html=True)
        render_image_card(
            primary_result["processed_image"],
            "Imagem selecionada",
            primary_result["file_name"],
            "preview-card",
        )
    else:
        st.markdown(
            """
            <div class="mobo-card empty-upload">
                <div class="empty-icon">+</div>
                <strong>Aguardando imagem</strong>
                <span>O resultado aparece ao lado assim que uma foto for enviada ou um exemplo for escolhido.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with result_col:
    if primary_result:
        render_detailed_result(primary_result)
    else:
        st.markdown(
            """
            <div class="mobo-card result-placeholder">
                <div class="card-kicker">Resultado da análise</div>
                <h2>Classe prevista</h2>
                <div class="confidence-placeholder">--.--%</div>
                <p>O painel mostrará maturação, confiança e mensagem explicativa sem alterar a resposta real do modelo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if analysis_results:
    if len(analysis_results) > 1:
        render_comparison_cards(analysis_results)
    render_detailed_visualization(primary_result)
    render_export_section(primary_result)

render_history_section()
