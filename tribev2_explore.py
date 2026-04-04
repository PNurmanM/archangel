"""
TRIBE v2 Explorer — Upload a video, get predicted brain activity.

TRIBE v2 is Meta's open-source brain encoding model. It predicts fMRI
brain responses (~20,484 cortical vertices on the fsaverage5 mesh)
from video, audio, or text stimuli.

Install:
  pip install "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
  pip install opencv-python nilearn
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ── Brain region atlas ──────────────────────────────────────────────
# Maps Destrieux atlas labels to: (readable name, category, what it does,
#   what activation might feel like / mean emotionally)

REGION_INFO = {
    # Visual cortex
    "G_cuneus": ("Cuneus", "visual", "Early visual processing — orientation, spatial frequency",
                 "Processing basic visual patterns, edges, and textures in the image"),
    "G_occipital_middle": ("Middle Occipital Gyrus", "visual", "Mid-level visual processing",
                           "Analyzing shapes and visual patterns"),
    "G_occipital_sup": ("Superior Occipital Gyrus", "visual", "Higher visual processing, spatial vision",
                        "Processing spatial layout and peripheral visual information"),
    "G_oc-temp_lat-fusifor": ("Fusiform Gyrus", "recognition", "Face and object recognition",
                               "Recognizing faces, bodies, words — the 'what is it?' region"),
    "G_oc-temp_med-Lingual": ("Lingual Gyrus", "visual", "Color perception, visual memory",
                               "Processing colors and recalling visual memories"),
    "G_oc-temp_med-Parahip": ("Parahippocampal Gyrus", "memory", "Scene recognition, spatial memory",
                               "Recognizing places and scenes — feeling of familiarity or novelty"),
    "Pole_occipital": ("Occipital Pole", "visual", "Primary visual cortex (V1)",
                       "The very first stage of seeing — raw visual input"),
    "S_calcarine": ("Calcarine Sulcus", "visual", "Primary visual cortex (V1) foveal vision",
                    "Processing what you're looking directly at"),
    "G_and_S_occipital_inf": ("Inferior Occipital Area", "visual", "Early visual processing",
                               "Processing visual details in the lower visual field"),
    "S_oc_middle_and_Lunatus": ("Middle Occipital Sulcus", "visual", "Visual association area",
                                 "Connecting visual features into coherent objects"),
    "S_oc_sup_and_transversal": ("Superior Occipital Sulcus", "visual", "Dorsal visual stream",
                                  "Processing motion and spatial relationships"),
    "S_occipital_ant": ("Anterior Occipital Sulcus", "visual", "Visual-temporal transition",
                         "Bridging basic vision with object recognition"),
    "S_oc-temp_lat": ("Lateral Occipitotemporal Sulcus", "recognition", "Object recognition pathway",
                       "Identifying and categorizing what you see"),
    "S_oc-temp_med_and_Lingual": ("Medial Occipitotemporal Sulcus", "visual", "Visual scene processing",
                                   "Processing scene layout and spatial context"),
    "S_collat_transv_post": ("Posterior Collateral Sulcus", "visual", "Visual association",
                              "Integrating visual features"),
    "S_collat_transv_ant": ("Anterior Collateral Sulcus", "memory", "Memory encoding area",
                             "Encoding new visual memories"),
    "S_parieto_occipital": ("Parieto-Occipital Sulcus", "visual", "Visual-spatial boundary",
                             "Transitioning from seeing to spatial understanding"),

    # Attention & spatial
    "G_parietal_sup": ("Superior Parietal Lobule", "attention", "Spatial attention, mental rotation",
                       "Focused attention, spatial awareness, mentally manipulating objects"),
    "G_pariet_inf-Angular": ("Angular Gyrus", "cognition", "Semantic understanding, reading, math",
                              "Making meaning — connecting words, numbers, and concepts"),
    "G_pariet_inf-Supramar": ("Supramarginal Gyrus", "cognition", "Phonological processing, empathy",
                               "Understanding language sounds, feeling empathy for others"),
    "G_precuneus": ("Precuneus", "self", "Self-reflection, mental imagery, consciousness",
                     "Daydreaming, imagining, self-awareness, autobiographical thinking"),
    "S_intrapariet_and_P_trans": ("Intraparietal Sulcus", "attention", "Attention control, numerosity",
                                   "Directing attention, estimating quantities"),
    "S_subparietal": ("Subparietal Sulcus", "self", "Default mode network",
                       "Mind-wandering, internal thought, self-referential thinking"),

    # Motor
    "G_precentral": ("Primary Motor Cortex", "motor", "Voluntary movement planning",
                     "Urge to move, action preparation, motor imagery"),
    "G_and_S_paracentral": ("Paracentral Lobule", "motor", "Leg/foot motor and sensory",
                             "Lower limb movement and sensation"),
    "S_precentral-inf-part": ("Inferior Precentral Sulcus", "motor", "Fine motor control",
                               "Hand and face movement control"),
    "S_precentral-sup-part": ("Superior Precentral Sulcus", "motor", "Gross motor planning",
                               "Planning large body movements"),
    "S_central": ("Central Sulcus", "motor", "Motor-sensory boundary",
                   "Border between movement and touch processing"),

    # Somatosensory
    "G_postcentral": ("Primary Somatosensory Cortex", "sensory", "Touch, pressure, temperature",
                       "Physical sensation — feeling of touch, texture, temperature"),
    "S_postcentral": ("Postcentral Sulcus", "sensory", "Somatosensory association",
                       "Integrating touch information from different body parts"),
    "G_and_S_subcentral": ("Subcentral Area", "sensory", "Taste and oral sensation",
                            "Taste processing, mouth and tongue sensation"),

    # Language & auditory
    "G_temp_sup-G_T_transv": ("Heschl's Gyrus", "auditory", "Primary auditory cortex",
                               "First stage of hearing — processing raw sound"),
    "G_temp_sup-Lateral": ("Superior Temporal Gyrus (lateral)", "language", "Speech comprehension (Wernicke's)",
                            "Understanding spoken language, processing speech"),
    "G_temp_sup-Plan_polar": ("Planum Polare", "auditory", "Auditory association",
                               "Processing complex sounds, voice recognition"),
    "G_temp_sup-Plan_tempo": ("Planum Temporale", "language", "Language lateralization",
                               "Processing speech sounds, musical pitch"),
    "G_temporal_middle": ("Middle Temporal Gyrus", "language", "Semantic processing, word meaning",
                           "Understanding word meanings, accessing knowledge"),
    "G_temporal_inf": ("Inferior Temporal Gyrus", "recognition", "Visual object recognition, reading",
                        "Identifying objects, reading words, categorizing things"),
    "Pole_temporal": ("Temporal Pole", "emotion", "Social cognition, emotional memory",
                       "Emotional associations, social understanding, feeling connected to memories"),
    "S_temporal_sup": ("Superior Temporal Sulcus", "social", "Social perception, theory of mind",
                        "Reading social cues, understanding others' intentions and emotions"),
    "S_temporal_inf": ("Inferior Temporal Sulcus", "recognition", "Visual recognition pathway",
                        "Object and face recognition processing"),
    "S_temporal_transverse": ("Transverse Temporal Sulcus", "auditory", "Auditory processing",
                               "Processing sound features"),
    "G_front_inf-Opercular": ("Pars Opercularis (Broca's)", "language", "Speech production, grammar",
                               "Producing speech, processing grammar and syntax"),
    "G_front_inf-Triangul": ("Pars Triangularis (Broca's)", "language", "Semantic processing",
                              "Understanding meaning in language"),
    "G_front_inf-Orbital": ("Pars Orbitalis", "language", "Semantic and social judgment",
                             "Making judgments about meaning and social situations"),

    # Executive / frontal
    "G_front_sup": ("Superior Frontal Gyrus", "executive", "Executive control, self-awareness",
                     "Planning, decision-making, sense of self, willpower"),
    "G_front_middle": ("Middle Frontal Gyrus", "executive", "Working memory, cognitive flexibility",
                        "Holding information in mind, switching between tasks"),
    "G_and_S_transv_frontopol": ("Frontopolar Cortex", "executive", "Abstract thinking, multitasking",
                                  "Complex planning, thinking about the future, juggling goals"),
    "G_and_S_frontomargin": ("Frontomarginal Area", "executive", "Decision-making under uncertainty",
                              "Making choices when outcomes are unclear"),
    "S_front_sup": ("Superior Frontal Sulcus", "executive", "Cognitive control",
                     "Top-down attention, staying focused"),
    "S_front_middle": ("Middle Frontal Sulcus", "executive", "Working memory",
                        "Keeping things in mind while thinking"),
    "S_front_inf": ("Inferior Frontal Sulcus", "executive", "Cognitive control",
                     "Inhibiting impulses, controlling behavior"),
    "G_orbital": ("Orbital Gyrus", "emotion", "Reward and value processing",
                   "Evaluating rewards, pleasure, wanting, craving"),
    "G_rectus": ("Gyrus Rectus", "emotion", "Reward processing, social behavior",
                  "Processing rewards, social decision-making"),
    "G_subcallosal": ("Subcallosal Gyrus", "emotion", "Mood regulation",
                       "Deep emotional processing, mood, sadness/happiness"),
    "S_orbital_lateral": ("Lateral Orbital Sulcus", "emotion", "Value judgment",
                           "Weighing pros and cons, evaluating options"),
    "S_orbital_med-olfact": ("Medial Orbital / Olfactory Sulcus", "emotion", "Smell and reward",
                              "Processing smells, reward anticipation"),
    "S_orbital-H_Shaped": ("H-Shaped Orbital Sulcus", "emotion", "Emotional evaluation",
                            "Evaluating emotional significance of stimuli"),
    "S_suborbital": ("Suborbital Sulcus", "emotion", "Emotional processing",
                      "Processing emotional content"),

    # Emotion / limbic / cingulate
    "G_and_S_cingul-Ant": ("Anterior Cingulate (ACC)", "emotion", "Emotion regulation, conflict monitoring",
                            "Detecting conflicts, regulating emotions, empathy, pain processing"),
    "G_and_S_cingul-Mid-Ant": ("Mid-Anterior Cingulate", "emotion", "Cognitive control, motivation",
                                "Motivation, effort, willingness to act"),
    "G_and_S_cingul-Mid-Post": ("Mid-Posterior Cingulate", "emotion", "Self-relevant processing",
                                 "Processing personally relevant information"),
    "G_cingul-Post-dorsal": ("Posterior Cingulate (dorsal)", "self", "Self-reflection, memory retrieval",
                              "Thinking about yourself, retrieving memories, mind-wandering"),
    "G_cingul-Post-ventral": ("Posterior Cingulate (ventral)", "self", "Default mode network hub",
                               "Daydreaming, internal narrative, autobiographical memory"),
    "S_cingul-Marginalis": ("Cingulate Marginal Sulcus", "emotion", "Cognitive-emotional integration",
                             "Connecting thoughts with feelings"),
    "S_pericallosal": ("Pericallosal Sulcus", "emotion", "Interhemispheric emotion processing",
                        "Coordinating emotional processing between brain halves"),
    "G_Ins_lg_and_S_cent_ins": ("Long Insular Gyrus", "emotion", "Interoception, emotional awareness",
                                 "Body awareness, gut feelings, emotional intensity"),
    "G_insular_short": ("Short Insular Gyrus", "emotion", "Disgust, empathy, body awareness",
                         "Visceral reactions, feeling disgust, empathizing with others' pain"),
    "S_circular_insula_ant": ("Anterior Circular Insular Sulcus", "emotion", "Emotional anticipation",
                               "Anticipating emotional events, anxiety, excitement"),
    "S_circular_insula_inf": ("Inferior Circular Insular Sulcus", "emotion", "Interoception",
                               "Sensing internal body states — heartbeat, breathing, hunger"),
    "S_circular_insula_sup": ("Superior Circular Insular Sulcus", "emotion", "Emotional-sensory integration",
                               "Connecting emotions with physical sensations"),

    # Other
    "Lat_Fis-ant-Horizont": ("Lateral Fissure (horizontal)", "language", "Language network boundary",
                              "Border region for language processing"),
    "Lat_Fis-ant-Vertical": ("Lateral Fissure (vertical)", "language", "Language area boundary",
                              "Border region near Broca's area"),
    "Lat_Fis-post": ("Posterior Lateral Fissure", "language", "Wernicke's area boundary",
                      "Border region for speech comprehension"),
    "S_interm_prim-Jensen": ("Intermediate Sulcus (Jensen)", "attention", "Attention network",
                              "Part of the attention control system"),
}

# Maps categories to emotional/cognitive themes for the final interpretation
CATEGORY_INFO = {
    "visual":      ("Visual Processing",                "seeing, analyzing visual details"),
    "recognition": ("Object/Face Recognition",          "identifying what's in the video"),
    "attention":   ("Attention & Spatial Awareness",     "focusing, spatial reasoning"),
    "motor":       ("Motor System",                      "urge to move, action planning"),
    "sensory":     ("Touch & Physical Sensation",        "bodily sensation, touch"),
    "auditory":    ("Auditory Processing",               "sound processing"),
    "language":    ("Language & Communication",           "inner speech, naming, meaning-making"),
    "executive":   ("Executive Function",                "planning, decision-making, focus"),
    "emotion":     ("Emotion & Feeling",                 "emotional reaction, gut feelings, mood"),
    "self":        ("Self-Referential Thinking",          "self-reflection, daydreaming, personal relevance"),
    "social":      ("Social Cognition",                   "reading social cues, empathy, understanding others"),
    "memory":      ("Memory",                             "remembering, familiarity, déjà vu"),
    "cognition":   ("Higher Cognition",                   "understanding, meaning, abstract thought"),
}


_CACHED_ATLAS = None


def get_brain_summary(preds: np.ndarray) -> str:
    """Map vertex activations to named brain regions and generate a human-readable summary."""
    global _CACHED_ATLAS
    from nilearn import datasets

    # Load the Destrieux atlas for fsaverage5 (cached after first call)
    if _CACHED_ATLAS is None:
        _CACHED_ATLAS = datasets.fetch_atlas_surf_destrieux(verbose=0)
    atlas = _CACHED_ATLAS
    labels_lh = atlas["map_left"]
    labels_rh = atlas["map_right"]
    label_names = atlas["labels"]

    # Combine left + right hemisphere labels
    all_labels = np.concatenate([labels_lh, labels_rh])

    # Average across timesteps
    mean_preds = preds.mean(axis=0)

    # Compute activation per region
    region_data = {}
    for idx, name in enumerate(label_names):
        if name in ("Unknown", "Medial_wall"):
            continue
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        vals = mean_preds[mask]
        region_data[name] = {
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "min": float(vals.min()),
            "std": float(vals.std()),
            "n_vertices": int(mask.sum()),
        }

    sorted_regions = sorted(region_data.items(), key=lambda x: abs(x[1]["mean"]), reverse=True)

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("BRAIN ACTIVITY SUMMARY")
    lines.append("=" * 60)

    # ── Top activated regions ──
    lines.append("\n  TOP 15 MOST ACTIVE BRAIN REGIONS:")
    lines.append(f"  {'#':<4} {'Region':<40} {'Score':>8}  What it does")
    lines.append("  " + "-" * 90)

    for i, (region, data) in enumerate(sorted_regions[:15], 1):
        info = REGION_INFO.get(region)
        if info:
            display_name, _, function, _ = info
        else:
            display_name = region
            function = ""
        sign = "+" if data["mean"] >= 0 else ""
        lines.append(f"  {i:<4} {display_name:<40} {sign}{data['mean']:.4f}  {function}")

    # ── Activity by brain system ──
    lines.append("\n  ACTIVITY BY BRAIN SYSTEM:")
    lines.append("  " + "-" * 70)

    category_scores = {}
    category_details = {}
    for region, data in region_data.items():
        info = REGION_INFO.get(region)
        cat = info[1] if info else "other"
        if cat not in category_scores:
            category_scores[cat] = []
            category_details[cat] = []
        category_scores[cat].append(data["mean"])
        if info:
            category_details[cat].append((info[0], data["mean"]))

    sorted_cats = sorted(category_scores.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)

    for cat, scores in sorted_cats:
        mean = np.mean(scores)
        cat_info = CATEGORY_INFO.get(cat, (cat, ""))
        label = cat_info[0]

        bar_len = min(int(abs(mean) * 300), 25)
        sign = "+" if mean >= 0 else "-"
        bar = "#" * max(bar_len, 1)
        state = "ACTIVE" if mean > 0.005 else "BELOW BASE" if mean < -0.005 else "BASELINE"

        lines.append(f"    {label:<35} {sign}{abs(mean):.4f}  [{state:>10}]  {bar}")

    # ── Emotional / cognitive profile ──
    # NOTE: We only report what IS active (positive scores). Negative scores
    # just mean "below baseline" — the brain is prioritizing other regions,
    # NOT that something bad is happening. We don't make negative inferences.
    lines.append("\n  EMOTIONAL & COGNITIVE PROFILE:")
    lines.append("  " + "-" * 70)

    emotion_indicators = []

    def _region_mean(names):
        """Get mean activation for a list of region names (only those present)."""
        vals = [region_data[r]["mean"] for r in names if r in region_data]
        return np.mean(vals) if vals else 0.0

    # Insula = emotional intensity, gut feelings
    insula_mean = _region_mean([
        "G_Ins_lg_and_S_cent_ins", "G_insular_short",
        "S_circular_insula_ant", "S_circular_insula_inf", "S_circular_insula_sup",
    ])
    if insula_mean > 0.01:
        emotion_indicators.append(("Emotional Arousal", "HIGH",
            "Insula active — strong gut reaction, visceral emotional response"))
    elif insula_mean > 0.005:
        emotion_indicators.append(("Emotional Arousal", "MILD",
            "Insula slightly active — some emotional engagement"))

    # Fusiform = face/body processing
    fusiform_data = region_data.get("G_oc-temp_lat-fusifor")
    if fusiform_data and fusiform_data["mean"] > 0.02:
        emotion_indicators.append(("Face/Body Detection", "STRONG",
            "Fusiform gyrus active — brain is recognizing faces or human forms"))
    elif fusiform_data and fusiform_data["mean"] > 0.005:
        emotion_indicators.append(("Face/Body Detection", "MILD",
            "Fusiform gyrus slightly active — possible face-like or body-like features"))

    # Limbic salience (temporal pole, subcallosal, ACC) — only report if active
    salience_mean = _region_mean(["Pole_temporal", "G_subcallosal", "G_and_S_cingul-Ant"])
    if salience_mean > 0.015:
        emotion_indicators.append(("Emotional Salience", "HIGH",
            "Limbic areas active — video is emotionally charged or highly meaningful"))
    elif salience_mean > 0.005:
        emotion_indicators.append(("Emotional Salience", "MODERATE",
            "Some limbic activation — video carries emotional weight"))

    # Reward circuit — only report if active
    reward_mean = _region_mean(["G_orbital", "G_rectus", "S_orbital-H_Shaped"])
    if reward_mean > 0.01:
        emotion_indicators.append(("Reward/Pleasure", "ACTIVE",
            "Orbitofrontal areas active — video feels pleasant, appealing, or rewarding"))

    # Social cognition (STS, temporal pole, supramarginal)
    social_mean = _region_mean(["S_temporal_sup", "Pole_temporal", "G_pariet_inf-Supramar"])
    if social_mean > 0.01:
        emotion_indicators.append(("Social Processing", "ACTIVE",
            "Social brain areas active — video involves people, social cues, or empathy"))
    elif social_mean > 0.005:
        emotion_indicators.append(("Social Processing", "MILD",
            "Some social brain activation — video may contain social elements"))

    # Self-referential (precuneus, PCC)
    self_mean = _region_mean(["G_precuneus", "G_cingul-Post-dorsal", "G_cingul-Post-ventral"])
    if self_mean > 0.01:
        emotion_indicators.append(("Self-Reflection", "ACTIVE",
            "Default mode network active — video triggers personal memories or self-referential thinking"))

    # Memory / familiarity
    memory_mean = _region_mean(["G_oc-temp_med-Parahip", "S_collat_transv_ant"])
    if memory_mean > 0.01:
        emotion_indicators.append(("Memory/Familiarity", "ACTIVE",
            "Parahippocampal areas active — video feels familiar or triggers scene/place memories"))

    # Attention engagement
    attention_mean = _region_mean(["G_parietal_sup", "S_intrapariet_and_P_trans", "G_front_middle"])
    if attention_mean > 0.01:
        emotion_indicators.append(("Attention Engagement", "HIGH",
            "Parietal-frontal network active — video grabs attention, visually complex"))

    # Motor activation
    motor_mean = _region_mean(["G_precentral", "G_and_S_paracentral"])
    if motor_mean > 0.01:
        emotion_indicators.append(("Motor Response", "ACTIVE",
            "Motor cortex active — video depicts action or triggers urge to move/respond"))

    # Empathy (supramarginal + STS + insula)
    empathy_mean = _region_mean(["G_pariet_inf-Supramar", "S_temporal_sup", "G_insular_short"])
    if empathy_mean > 0.01:
        emotion_indicators.append(("Empathy", "ACTIVE",
            "Empathy network active — brain is simulating or relating to others' emotions"))

    # Fear/anxiety (ACC + anterior insula)
    anxiety_mean = _region_mean(["G_and_S_cingul-Ant", "S_circular_insula_ant"])
    if anxiety_mean > 0.02:
        emotion_indicators.append(("Alertness/Anxiety", "HIGH",
            "ACC + anterior insula active — heightened alertness or unease"))

    if emotion_indicators:
        for label, level, desc in emotion_indicators:
            lines.append(f"    [{level:>10}]  {label}: {desc}")
    else:
        lines.append("    No strong emotional or cognitive signals detected.")
        lines.append("    The brain response is primarily perceptual (seeing) rather than emotional.")

    # ── What the brain is doing (narrative) ──
    lines.append("\n  WHAT YOUR BRAIN IS DOING WITH THIS VIDEO:")
    lines.append("  " + "-" * 70)

    top3_cats = sorted_cats[:3]
    for cat, scores in top3_cats:
        mean = np.mean(scores)
        if abs(mean) < 0.003:
            continue
        cat_info = CATEGORY_INFO.get(cat, (cat, cat))
        state = "engaging" if mean > 0 else "suppressing"

        # Get the top contributing region in this category
        details = category_details.get(cat, [])
        if details:
            top_region = sorted(details, key=lambda x: abs(x[1]), reverse=True)[0]
            region_info = None
            for rname, rdata in REGION_INFO.items():
                if rdata[0] == top_region[0]:
                    region_info = rdata
                    break
            if region_info:
                lines.append(f"    The brain is {state} {cat_info[1]}.")
                lines.append(f"      Strongest contributor: {region_info[0]}")
                lines.append(f"      This means: {region_info[3]}")
                lines.append("")

    # ── Overall intensity ──
    overall = np.mean([abs(d["mean"]) for d in region_data.values()])
    lines.append("  OVERALL BRAIN ENGAGEMENT:")
    lines.append("  " + "-" * 70)
    if overall > 0.05:
        lines.append("    VERY HIGH — This video strongly activates the brain.")
    elif overall > 0.03:
        lines.append("    HIGH — This video produces significant brain activity.")
    elif overall > 0.015:
        lines.append("    MODERATE — This video produces a noticeable brain response.")
    elif overall > 0.005:
        lines.append("    MILD — This video produces a subtle brain response.")
    else:
        lines.append("    MINIMAL — This video produces very little brain response.")
    lines.append(f"    (Average absolute activation: {overall:.4f})")

    lines.append("")
    return "\n".join(lines)


# ── Input detection ────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def is_video(file_path: str) -> bool:
    """Check if a file is a video based on its extension."""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


# ── Model loading (separated so it can be done once) ──────────────

def load_model(cache_folder: str = "./cache"):
    """Load the TRIBE v2 model. Call once, reuse for many predictions."""
    t_import = time.time()
    from tribev2.demo_utils import TribeModel
    dt_import = time.time() - t_import
    print(f"  Imports ready in {dt_import:.1f}s")

    cache = Path(cache_folder)
    cache.mkdir(exist_ok=True)

    print("  Loading TRIBE v2 model...")
    t0 = time.time()
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(cache))
    dt = time.time() - t0
    print(f"  Model loaded in {dt:.1f}s")
    return model


# ── Main pipeline ──────────────────────────────────────────────────

def run_tribev2(input_path: str, model=None, cache_folder: str = "./cache",
                output_dir: str = "./outputs", visualize: bool = False,
                save_frames: bool = False, make_movie: bool = True,
                movie_fps: int = 2, surface: str = "inflated",
                html: bool = False):
    """Process a video and predict brain activity.

    If `model` is provided, skips loading (fast). Otherwise loads fresh (slow).
    """
    input_stem = Path(input_path).stem
    out_dir = Path(output_dir) / input_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    step_total = 6 if visualize else 5
    t_start = time.time()

    # --- 1. Load model (skip if already loaded) ---
    if model is None:
        print(f"\n[1/{step_total}] Loading TRIBE v2 model (first run is slow)...")
        model = load_model(cache_folder)
    else:
        print(f"\n[1/{step_total}] Model already loaded — skipping.")

    # --- 2. Validate input ---
    if not is_video(input_path):
        ext = Path(input_path).suffix
        print(f"Error: Unsupported file type '{ext}'")
        print(f"  Supported videos: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(1)
    print(f"\n[2/{step_total}] Video input: {input_path}")

    # --- 3. Build events and predict ---
    print(f"\n[3/{step_total}] Building events dataframe...")
    import pandas as pd
    from tribev2.demo_utils import get_audio_and_text_events

    video_event = {
        "type": "Video",
        "filepath": str(Path(input_path).resolve()),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    df = get_audio_and_text_events(pd.DataFrame([video_event]), audio_only=True)
    print("      Events dataframe:")
    print(df[["type", "start", "duration"]].to_string(index=False))

    print(f"\n[4/{step_total}] Running brain activity prediction...")
    preds, segments = model.predict(events=df)

    # --- 4. Raw results ---
    print("\n" + "=" * 60)
    print("RAW RESULTS")
    print("=" * 60)
    print(f"  Input file        : {input_path}")
    print(f"  Predictions shape : {preds.shape}")
    print(f"    - {preds.shape[0]} timestep(s)")
    print(f"    - {preds.shape[1]} cortical vertices (fsaverage5 mesh)")
    print(f"  Value range       : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"  Mean activation   : {preds.mean():.4f}")

    # --- 5. Human-readable summary ---
    print(f"\n[5/{step_total}] Mapping vertices to brain regions...")
    summary = get_brain_summary(preds)
    print(summary)

    # Save raw predictions to output folder
    npy_path = out_dir / f"{input_stem}_brain_preds.npy"
    np.save(str(npy_path), preds)
    print(f"  Raw predictions saved to: {npy_path}")

    # Save summary to text file
    summary_path = out_dir / f"{input_stem}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Summary saved to: {summary_path}")

    # --- 6. Brain visualization ---
    if visualize:
        print(f"\n[6/{step_total}] Generating brain visualizations...")
        try:
            from brain_visualizer import visualize_all
            visualize_all(
                preds=preds,
                output_dir=str(out_dir),
                image_stem=input_stem,
                surface=surface,
                save_frames=save_frames,
                make_movie=make_movie,
                movie_fps=movie_fps,
                html=html,
            )
        except ImportError as e:
            print(f"  [!] Visualization failed — missing dependency: {e}")
            print("      Run: pip install nilearn matplotlib")
        except Exception as e:
            print(f"  [!] Visualization failed: {e}")

    dt = time.time() - t_start
    print(f"\n  All outputs saved to: {out_dir}")
    print(f"  Total time: {dt:.1f}s")
    return preds, segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRIBE v2 Brain Activity Explorer")
    parser.add_argument("input", nargs="?", help="Path to a video file")
    parser.add_argument("--cache", default="./cache", help="Cache folder for model weights")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--visualize", action="store_true", help="Generate brain surface visualizations (PNG + movie)")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frame PNGs per timestep")
    parser.add_argument("--html", action="store_true",
                        help="Generate interactive HTML brain viewer (near-instant, opens in browser)")
    parser.add_argument("--no-movie", action="store_true", help="Skip movie generation")
    parser.add_argument("--movie-fps", type=int, default=2, help="Movie frames per second (default: 2)")
    parser.add_argument("--surface", default="inflated", choices=["inflated", "pial", "white"],
                        help="Brain surface type (default: inflated)")
    parser.add_argument("--serve", action="store_true",
                        help="Server mode: load model once, then process files interactively")
    args = parser.parse_args()

    if args.serve:
        # ── Server mode: load once, run many ──
        print("=" * 60)
        print("TRIBE v2 — SERVER MODE")
        print("=" * 60)
        print("Loading model once...")
        model = load_model(args.cache)
        print("\nReady! Paste a video path and press Enter.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down.")
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Shutting down.")
                break
            # Strip quotes (drag-and-drop often adds them)
            file_path = user_input.strip("\"'")
            if not Path(file_path).exists():
                print(f"  File not found: {file_path}")
                continue
            if not is_video(file_path):
                ext = Path(file_path).suffix
                print(f"  Unsupported file type: {ext}")
                continue
            try:
                run_tribev2(
                    file_path,
                    model=model,
                    cache_folder=args.cache,
                    output_dir=args.output_dir,
                    visualize=args.visualize,
                    save_frames=args.save_frames,
                    make_movie=not args.no_movie,
                    movie_fps=args.movie_fps,
                    surface=args.surface,
                    html=args.html,
                )
            except Exception as e:
                print(f"  Error: {e}")
            print()
    else:
        # ── Single-file mode ──
        if not args.input:
            parser.error("the following arguments are required: input (or use --serve)")
        if not Path(args.input).exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)

        preds, segments = run_tribev2(
            args.input,
            cache_folder=args.cache,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_frames=args.save_frames,
            make_movie=not args.no_movie,
            movie_fps=args.movie_fps,
            surface=args.surface,
            html=args.html,
        )
