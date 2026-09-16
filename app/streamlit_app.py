import streamlit as st
import yaml
import psutil
import torch
from pathlib import Path
from app.model_loader import load_config, initialize_model, get_model_info, check_adapter_exists

st.set_page_config(page_title="Indian Legal QA", layout="wide")

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

@st.cache_resource
def init_model():
    config = load_config()
    use_lora = check_adapter_exists(config["training"]["output_dir"])
    generator, model, tokenizer = initialize_model(
        config["model"]["base_model_id"],
        use_lora=use_lora,
        adapter_path=config["training"]["output_dir"] if use_lora else None
    )
    return generator, config

st.title("🏛️ Indian Legal Question Answering System")
st.markdown("""
⚠️ **Educational Disclaimer**: This application is for educational and demonstration purposes only.  
It does not provide legal advice and should not be relied upon for legal decisions.  
Always consult qualified legal professionals for actual legal matters.
""")

generator, config = init_model()

with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_info = get_model_info(config)
    st.subheader("Model Status")
    st.write(f"**Base Model**: {model_info['base_model']}")
    st.write(f"**Mode**: {model_info['mode']}")
    
    if model_info['adapter_exists']:
        st.success("✅ Fine-tuned adapter loaded")
    else:
        st.warning("⚠️ Base model only (no fine-tuned adapter)")
    
    st.divider()
    
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.1, 2.0, config["generation"]["temperature"])
    max_tokens = st.slider("Max New Tokens", 100, 1024, config["generation"]["max_new_tokens"])
    top_p = st.slider("Top-P", 0.1, 1.0, config["generation"]["top_p"])
    
    st.divider()
    
    st.subheader("System Info")
    if torch.cuda.is_available():
        st.write(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        st.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        st.write("CPU only (no GPU detected)")
    
    st.divider()
    
    with st.expander("📋 Fine-Tuning Details"):
        st.write(f"**Dataset**: {config['dataset']['name']}")
        st.write(f"**LoRA Rank**: {config['lora']['r']}")
        st.write(f"**LoRA Alpha**: {config['lora']['lora_alpha']}")
        st.write(f"**Target Modules**: {', '.join(config['lora']['target_modules'])}")
        st.write(f"**Training Epochs**: {config['training']['num_epochs']}")
        st.write(f"**Learning Rate**: {config['training']['learning_rate']}")
        st.write(f"**Batch Size (eff)**: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
        st.write(f"**Quantization**: 4-bit NF4")
        
        if not model_info['adapter_exists']:
            st.info("✏️ **Fine-tuned adapter not generated yet.**  The complete training pipeline is ready.  Run `python src/train/train_lora.py` to generate the adapter.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a Legal Question")
    question = st.text_area(
        "Enter your question about Indian law:",
        height=100,
        placeholder="e.g., What is Section 420 of the Indian Penal Code?"
    )

with col2:
    st.subheader("Examples")
    examples = [
        "What is the Indian Penal Code?",
        "Explain Section 420 IPC",
        "What are fundamental rights?",
        "Define bail in CrPC"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            question = ex
            st.rerun()

if question:
    with st.spinner("🔄 Generating response..."):
        try:
            answer = generator.generate(
                question,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p
            )
            
            st.subheader("Answer")
            st.markdown(answer)
            
            st.divider()
            st.caption(f"Model: {model_info['mode']} | Temp: {temperature} | Max tokens: {max_tokens}")
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")