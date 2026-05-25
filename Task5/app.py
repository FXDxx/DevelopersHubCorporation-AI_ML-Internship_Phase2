import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report
import io

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------- #
# Page Configuration
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .dark-theme .prediction-card {
        background-color: #262730;
    }
    .highlight-word {
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
        margin: 0 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# Constants & Configuration
# ---------------------------------------------------------------------------- #
LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
LABEL_EMOJI = {'World': '🌍', 'Sports': '⚽', 'Business': '💼', 'Sci/Tech': '🔬'}
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

# ---------------------------------------------------------------------------- #
# Sidebar & Model Loading
# ---------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

device_option = st.sidebar.selectbox(
    "Compute Device",
    options=["Auto", "CPU"],
    index=0,
    help="Auto will use CUDA if available."
)
DEVICE = 0 if (device_option == "Auto" and torch.cuda.is_available()) else -1
DEVICE_STR = 'cuda' if DEVICE == 0 else 'cpu'

st.sidebar.markdown("### Model Source")
model_source_type = st.sidebar.radio("Select Source", ["Local Model", "Hugging Face Hub"])

if model_source_type == "Local Model":
    model_path = st.sidebar.text_input("Local Directory Path", value="./bert_ag_news_model")
else:
    model_path = st.sidebar.text_input("Hugging Face Hub Model ID", value="textattack/bert-base-uncased-ag-news")

@st.cache_resource(show_spinner=False)
def load_model(path, device):
    try:
        classifier = hf_pipeline(
            'text-classification', 
            model=path, 
            tokenizer=path, 
            device=device,
            top_k=None, # Get all class probabilities
            truncation=True,
            max_length=128
        )
        # Try to load tokenizer separately for word importance if needed
        tokenizer = AutoTokenizer.from_pretrained(path)
        return classifier, tokenizer, None
    except Exception as e:
        return None, None, str(e)

with st.sidebar:
    st.markdown("---")
    with st.spinner("Loading Model..."):
        classifier, tokenizer, error_msg = load_model(model_path, DEVICE)
    
    if error_msg:
        st.error(f"Failed to load model:\\n{error_msg}")
    else:
        st.success(f"Model loaded successfully on {DEVICE_STR.upper()}!")

# ---------------------------------------------------------------------------- #
# Main Layout - Tabs
# ---------------------------------------------------------------------------- #
st.title("📰 News Topic Classifier (BERT)")
st.markdown("A professional UI for categorizing news headlines into **World, Sports, Business, and Sci/Tech**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Prediction", 
    "📁 Batch Classification", 
    "📊 Training Diagnostics", 
    "🔍 Dataset Explorer"
])

# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #
def compute_word_importance(text, classifier, tokenizer):
    """Computes word importance by masking each word and observing probability drop."""
    words = text.split()
    if not words: return []
    
    # Baseline
    base_res = classifier(text)[0]
    base_res_dict = {res['label']: res['score'] for res in base_res}
    
    # Try to map labels to our standard labels if possible
    # TextAttack uses LABEL_0 etc. We need to handle this.
    target_label = max(base_res, key=lambda x: x['score'])['label']
    base_prob = base_res_dict[target_label]

    importances = []
    
    # Mask each word
    for i in range(len(words)):
        masked_words = words.copy()
        masked_words[i] = "[MASK]"
        masked_text = " ".join(masked_words)
        
        try:
            res = classifier(masked_text)[0]
            res_dict = {r['label']: r['score'] for r in res}
            masked_prob = res_dict.get(target_label, 0.0)
            
            # Importance = Drop in probability
            importance = base_prob - masked_prob
            importances.append(importance)
        except:
            importances.append(0.0)
            
    # Normalize
    max_imp = max(abs(x) for x in importances) if importances else 1.0
    if max_imp == 0: max_imp = 1.0
    norm_importances = [imp / max_imp for imp in importances]
    
    return list(zip(words, norm_importances))

def get_standard_label(raw_label):
    """Maps arbitrary label formats to standard AG News labels."""
    l = str(raw_label).lower()
    if "0" in l or l == "world": return "World"
    if "1" in l or l == "sports": return "Sports"
    if "2" in l or l == "business": return "Business"
    if "3" in l or l == "sci/tech" or "sci" in l: return "Sci/Tech"
    return "Unknown"

# ---------------------------------------------------------------------------- #
# Tab 1: Single Prediction
# ---------------------------------------------------------------------------- #
with tab1:
    st.markdown("### Enter a News Headline")
    
    # Default examples
    examples = [
        "NASA discovers liquid water on a distant exoplanet",
        "Global stock markets rally after Federal Reserve announcement",
        "Lionel Messi scores a spectacular hat-trick in the final",
        "UN Security Council meets to discuss international peace"
    ]
    
    selected_example = st.selectbox("Or choose an example:", [""] + examples)
    
    user_input = st.text_area("Headline:", value=selected_example, height=100, placeholder="Type or paste a news headline here...")
    
    if st.button("Predict Topic", type="primary", use_container_width=True):
        if not classifier:
            st.error("Model is not loaded. Please check configuration.")
        elif not user_input.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing..."):
                raw_results = classifier(user_input)[0]
                
                # Standardize results
                std_results = {get_standard_label(r['label']): r['score'] for r in raw_results}
                
                # Sort by score
                sorted_res = sorted(std_results.items(), key=lambda x: x[1], reverse=True)
                top_label = sorted_res[0][0]
                top_score = sorted_res[0][1]
                top_emoji = LABEL_EMOJI.get(top_label, '✨')
                
                # Layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="margin:0; color:#555;">Predicted Topic</h4>
                        <h1 style="margin:10px 0; font-size: 3rem;">{top_emoji} {top_label}</h1>
                        <h3 style="margin:0; color:#888;">{top_score:.2%} Confidence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    # Bar Chart
                    df_chart = pd.DataFrame(sorted_res, columns=['Topic', 'Confidence'])
                    fig = px.bar(
                        df_chart, 
                        x='Confidence', 
                        y='Topic', 
                        orientation='h',
                        color='Topic',
                        color_discrete_map={k: v for k, v in zip(LABEL_NAMES, COLORS)},
                        text_auto='.1%'
                    )
                    fig.update_layout(
                        xaxis_tickformat='%', 
                        showlegend=False, 
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=200,
                        yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Word Importance
                st.markdown("### AI Word Attribution")
                st.markdown("Words highlighted in **<span style='color:green;'>green</span>** contributed most to the prediction. Words in **<span style='color:red;'>red</span>** pulled the prediction towards other classes.", unsafe_allow_html=True)
                
                importances = compute_word_importance(user_input, classifier, tokenizer)
                
                html_words = []
                for word, imp in importances:
                    if imp > 0:
                        # Green scale
                        intensity = int(imp * 255)
                        bg_color = f"rgba(0, 255, 0, {imp * 0.4})"
                    elif imp < 0:
                        # Red scale
                        intensity = int(abs(imp) * 255)
                        bg_color = f"rgba(255, 0, 0, {abs(imp) * 0.4})"
                    else:
                        bg_color = "transparent"
                        
                    html_words.append(f"<span class='highlight-word' style='background-color: {bg_color};'>{word}</span>")
                
                st.markdown(f"<div style='font-size: 1.2rem; line-height: 1.8; padding: 20px; border: 1px solid #ddd; border-radius: 8px;'>{' '.join(html_words)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- #
# Tab 2: Batch Classification
# ---------------------------------------------------------------------------- #
with tab2:
    st.markdown("### Upload a file for batch processing")
    st.markdown("Upload a CSV file containing a column of news headlines.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_csv(uploaded_file, sep='\t', header=None, names=['text'])
                
            text_col = st.selectbox("Select the column containing the text:", df_batch.columns)
            
            if st.button("Start Batch Processing", type="primary"):
                if not classifier:
                    st.error("Model is not loaded.")
                else:
                    texts = df_batch[text_col].dropna().astype(str).tolist()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    confidences = []
                    
                    # Process in smaller batches
                    batch_size = 16
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        
                        # Note: we catch errors for empty strings
                        valid_texts = [t if t.strip() else "empty" for t in batch_texts]
                        
                        batch_results = classifier(valid_texts)
                        
                        for res in batch_results:
                            # res is a list of dicts because top_k=None
                            # sort it
                            sorted_res = sorted(res, key=lambda x: x['score'], reverse=True)
                            best = sorted_res[0]
                            predictions.append(get_standard_label(best['label']))
                            confidences.append(best['score'])
                            
                        # Update progress
                        progress = min((i + batch_size) / len(texts), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {min(i + batch_size, len(texts))} / {len(texts)}")
                        
                    df_batch['Predicted Topic'] = predictions
                    df_batch['Confidence'] = confidences
                    
                    st.success("Batch processing complete!")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(df_batch, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            df_batch, 
                            names='Predicted Topic', 
                            title='Topic Distribution',
                            color='Predicted Topic',
                            color_discrete_map={k: v for k, v in zip(LABEL_NAMES, COLORS)},
                            hole=0.4
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    # Download button
                    csv = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='classified_headlines.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------------------------------------------------------- #
# Tab 3: Training Diagnostics
# ---------------------------------------------------------------------------- #
with tab3:
    st.markdown("### Model Training Performance")
    
    # Simulate loading metrics if the actual results folder isn't fully structured for streamlit
    # We will try to read from trainer_state.json if it exists
    trainer_state_path = os.path.join(model_path, "trainer_state.json")
    
    has_real_data = False
    if os.path.exists(trainer_state_path):
        try:
            import json
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
            
            history = state.get('log_history', [])
            train_steps = []
            train_loss = []
            eval_steps = []
            eval_loss = []
            
            for entry in history:
                if 'loss' in entry and 'step' in entry:
                    train_steps.append(entry['step'])
                    train_loss.append(entry['loss'])
                if 'eval_loss' in entry and 'step' in entry:
                    eval_steps.append(entry['step'])
                    eval_loss.append(entry['eval_loss'])
                    
            if train_steps and eval_steps:
                has_real_data = True
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_steps, y=train_loss, mode='lines', name='Train Loss', line=dict(color='#4e79a7')))
                fig.add_trace(go.Scatter(x=eval_steps, y=eval_loss, mode='lines+markers', name='Eval Loss', line=dict(color='#e15759')))
                
                fig.update_layout(title="Training & Evaluation Loss", xaxis_title="Step", yaxis_title="Loss", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            pass
            
    if not has_real_data:
        st.info("No interactive training logs (`trainer_state.json`) found in the model directory. Displaying illustrative mock metrics for dashboard demonstration.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", "94.5%", "+1.2%")
        col2.metric("Macro F1", "0.942", "+0.01")
        col3.metric("Eval Loss", "0.231", "-0.05")
        col4.metric("Train Epochs", "3", None)
        
        # Mock confusion matrix
        st.markdown("#### Evaluation Confusion Matrix (Simulated)")
        cm = np.array([
            [1200, 20, 15, 15],
            [10, 1220, 10, 10],
            [30, 15, 1150, 55],
            [20, 5, 60, 1165]
        ])
        
        fig = px.imshow(
            cm, 
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=LABEL_NAMES,
            y=LABEL_NAMES,
            color_continuous_scale="Blues",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------- #
# Tab 4: Dataset Explorer
# ---------------------------------------------------------------------------- #
with tab4:
    st.markdown("### AG News Dataset Explorer")
    st.markdown("Browse raw samples from the Hugging Face `ag_news` dataset to understand the training distribution.")
    
    @st.cache_data(show_spinner=False)
    def fetch_dataset():
        try:
            ds = load_dataset('ag_news')
            df_train = ds['train'].to_pandas()
            # Map labels
            df_train['Label Name'] = df_train['label'].map(lambda x: LABEL_NAMES[x])
            return df_train
        except Exception as e:
            return str(e)
            
    with st.spinner("Fetching dataset..."):
        df_ds = fetch_dataset()
        
    if isinstance(df_ds, str):
        st.error(f"Failed to load dataset: {df_ds}")
    else:
        # Filtering
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("Filter by Category:", ["All"] + LABEL_NAMES)
        with col2:
            num_samples = st.slider("Number of samples to view:", 5, 50, 10)
            
        if selected_class != "All":
            df_view = df_ds[df_ds['Label Name'] == selected_class]
        else:
            df_view = df_ds
            
        st.dataframe(
            df_view[['Label Name', 'text']].sample(min(len(df_view), num_samples), random_state=42), 
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("#### Dataset Statistics")
        class_counts = df_ds['Label Name'].value_counts().reset_index()
        fig = px.bar(
            class_counts, 
            x='Label Name', 
            y='count', 
            title='Training Set Class Distribution',
            color='Label Name',
            color_discrete_map={k: v for k, v in zip(LABEL_NAMES, COLORS)}
        )
        st.plotly_chart(fig, use_container_width=True)
