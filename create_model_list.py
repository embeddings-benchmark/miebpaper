from __future__ import annotations

import logging

from mteb.model_meta import ModelMeta
from mteb.models import (
    align_models,
    arctic_models,
    bedrock_models,
    bge_models,
    blip2_models,
    blip_models,
    bm25,
    cde_models,
    clip_models,
    codesage_models,
    cohere_models,
    cohere_v,
    colbert_models,
    dino_models,
    e5_instruct,
    e5_models,
    e5_v,
    evaclip_models,
    fa_models,
    gme_v_models,
    google_models,
    gritlm_models,
    gte_models,
    ibm_granite_models,
    inf_models,
    jasper_models,
    jina_clip,
    jina_models,
    lens_models,
    linq_models,
    llm2vec_models,
    misc_models,
    moco_models,
    model2vec_models,
    moka_models,
    mxbai_models,
    no_instruct_sentence_models,
    nomic_models,
    nomic_models_vision,
    nvidia_models,
    openai_models,
    openclip_models,
    piccolo_models,
    promptriever_models,
    qodo_models,
    qtack_models,
    repllama_models,
    rerankers_custom,
    rerankers_monot5_based,
    ru_sentence_models,
    salesforce_models,
    sentence_transformers_models,
    siglip_models,
    sonar_models,
    stella_models,
    text2vec_models,
    uae_models,
    vista_models,
    vlm2vec_models,
    voyage_models,
    voyage_v,
)

logger = logging.getLogger(__name__)

model_modules = [
    align_models,
    arctic_models,
    bedrock_models,
    bge_models,
    blip2_models,
    blip_models,
    bm25,
    clip_models,
    codesage_models,
    cde_models,
    cohere_models,
    cohere_v,
    colbert_models,
    dino_models,
    e5_instruct,
    e5_models,
    e5_v,
    evaclip_models,
    google_models,
    gritlm_models,
    gte_models,
    ibm_granite_models,
    inf_models,
    jasper_models,
    jina_models,
    jina_clip,
    lens_models,
    linq_models,
    llm2vec_models,
    misc_models,
    model2vec_models,
    moka_models,
    moco_models,
    mxbai_models,
    no_instruct_sentence_models,
    nomic_models,
    nomic_models_vision,
    nvidia_models,
    openai_models,
    openclip_models,
    piccolo_models,
    gme_v_models,
    promptriever_models,
    qodo_models,
    qtack_models,
    repllama_models,
    rerankers_custom,
    rerankers_monot5_based,
    ru_sentence_models,
    salesforce_models,
    sentence_transformers_models,
    siglip_models,
    vista_models,
    vlm2vec_models,
    voyage_v,
    stella_models,
    sonar_models,
    text2vec_models,
    uae_models,
    voyage_models,
    fa_models,
]
MIEB_MODEL_REGISTRY = {}

for module in model_modules:
    for mdl in vars(module).values():
        if isinstance(mdl, ModelMeta):
            if "image" in mdl.modalities:
                MIEB_MODEL_REGISTRY[mdl.name] = mdl

all_mieb_model_names = MIEB_MODEL_REGISTRY.keys()
num_models = len(all_mieb_model_names)
logger.info(f"{num_models=}")

columns = ["Model Name", "Type", "Model Size", "Modalities"]

main_latex_table = """\\begin{table*}\centering
% \scriptsize
\centering
\\resizebox{0.7\\textwidth}{!}{
\\begin{tabular}{lccc}\\toprule\n"""

title_col = " &".join([f"\\textbf{{{c}}}" for c in columns])

main_latex_table += title_col
main_latex_table += " \\\\\midrule\n"

for model_name, meta in MIEB_MODEL_REGISTRY.items():
    model_type = "Encoder"
    for mllm_name in ["voyage", "vlm2vec", "e5"]:
        if mllm_name in meta.name.lower():
            model_type = "MLLM"
            break
    model_size = int(meta.n_parameters / 1e6) if meta.n_parameters else "N/A"
    modalities = ", ".join(meta.modalities)
    row = f"{model_name} & {model_type} & {model_size} & {modalities} \\\\ \n"
    main_latex_table += row


main_latex_table += """\\bottomrule
\end{tabular}}
\caption{List of all models evaluated in MIEB. Model sizes are in millions of parameters.}\label{tab: list of models}
\end{table*}"""

print(main_latex_table)
