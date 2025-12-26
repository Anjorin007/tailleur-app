import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math
import warnings
warnings.filterwarnings('ignore')

# Configuration page
st.set_page_config(
    page_title="Tailleur Intelligent",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé - Design moderne
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .header {
        background: linear-gradient(135deg, #0066cc 0%, #0099ff 50%, #00ccff 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 0;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 102, 204, 0.3);
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: #0066cc;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #0099ff;
        padding-bottom: 0.5rem;
    }
    
    .measurement-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #0099ff;
    }
    
    .measurement-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid #eee;
    }
    
    .measurement-row:last-child {
        border-bottom: none;
    }
    
    .measurement-label {
        font-size: 1rem;
        color: #333;
        font-weight: 500;
    }
    
    .measurement-value {
        font-size: 1.2rem;
        color: #0066cc;
        font-weight: bold;
    }
    
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #0066cc 0%, #0099ff 50%, #00ccff 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .upload-area {
        border: 2px dashed #0099ff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f0f8ff;
    }
    
    </style>
""", unsafe_allow_html=True)

# ==================== FONCTIONS ====================

@st.cache_resource
def load_yolo_model():
    """Charge YOLOv8 une seule fois"""
    return YOLO('yolov8n-pose.pt')

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def pixels_to_cm(pixel_distance, reference_height_cm, reference_height_px):
    if reference_height_px == 0:
        return 0
    return pixel_distance * (reference_height_cm / reference_height_px)

def estimate_depth_from_width(width_cm, body_part="chest"):
    ratios = {"chest": 0.73, "waist": 0.78, "hips": 0.88}
    return width_cm * ratios.get(body_part, 0.75)

def calculate_circumference_ellipse(width, depth):
    a, b = width / 2, depth / 2
    h = ((a - b)**2) / ((a + b)**2)
    return math.pi * (a + b) * (1 + (3*h) / (10 + math.sqrt(4 - 3*h)))

def check_keypoint_visibility(kp, confidence_threshold=0.3):
    if len(kp) < 3:
        return False
    return kp[0] > 0 and kp[1] > 0 and (len(kp) < 3 or kp[2] > confidence_threshold)

def detect_person_and_pose(image, model):
    results = model(image, verbose=False)
    
    if len(results[0].boxes) == 0:
        return None, None, image
    
    result = results[0]
    bbox = result.boxes.xyxy[0].cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None
    
    if keypoints is not None and kp_conf is not None:
        kp_with_conf = []
        for kp, conf in zip(keypoints, kp_conf):
            kp_full = np.concatenate([kp, conf.reshape(-1, 1)], axis=1)
            kp_with_conf.append(kp_full)
        keypoints = np.array(kp_with_conf)
    
    annotated = result.plot()
    return bbox, keypoints, annotated

def calculate_body_measurements(image, keypoints, reference_height_cm):
    if keypoints is None or len(keypoints) == 0:
        return None
    
    kp = keypoints[0]
    measurements = {}
    
    if check_keypoint_visibility(kp[0]):
        head_point = kp[0][:2]
        ankle_left = kp[15][:2] if check_keypoint_visibility(kp[15]) else None
        ankle_right = kp[16][:2] if check_keypoint_visibility(kp[16]) else None
        
        if ankle_left is not None or ankle_right is not None:
            if ankle_left is not None and ankle_right is not None:
                lowest_ankle = ankle_left if ankle_left[1] > ankle_right[1] else ankle_right
            else:
                lowest_ankle = ankle_left if ankle_left is not None else ankle_right
            
            height_px = calculate_distance(head_point, lowest_ankle)
            measurements['height_px'] = height_px
            measurements['height_total_cm'] = reference_height_cm
    
    if 'height_px' not in measurements:
        return None
    
    height_px = measurements['height_px']
    
    if check_keypoint_visibility(kp[5]) and check_keypoint_visibility(kp[9]):
        arm_left_px = calculate_distance(kp[5][:2], kp[9][:2])
        measurements['arm_length_left_cm'] = pixels_to_cm(arm_left_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[6]) and check_keypoint_visibility(kp[10]):
        arm_right_px = calculate_distance(kp[6][:2], kp[10][:2])
        measurements['arm_length_right_cm'] = pixels_to_cm(arm_right_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[11]) and check_keypoint_visibility(kp[15]):
        leg_left_px = calculate_distance(kp[11][:2], kp[15][:2])
        measurements['leg_length_left_cm'] = pixels_to_cm(leg_left_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[12]) and check_keypoint_visibility(kp[16]):
        leg_right_px = calculate_distance(kp[12][:2], kp[16][:2])
        measurements['leg_length_right_cm'] = pixels_to_cm(leg_right_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[5]) and check_keypoint_visibility(kp[6]):
        shoulder_width_px = calculate_distance(kp[5][:2], kp[6][:2])
        measurements['shoulder_width_cm'] = pixels_to_cm(shoulder_width_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[11]) and check_keypoint_visibility(kp[12]):
        hip_width_px = calculate_distance(kp[11][:2], kp[12][:2])
        measurements['hip_width_cm'] = pixels_to_cm(hip_width_px, reference_height_cm, height_px)
    
    if check_keypoint_visibility(kp[5]) and check_keypoint_visibility(kp[11]):
        torso_px = calculate_distance(kp[5][:2], kp[11][:2])
        measurements['torso_length_cm'] = pixels_to_cm(torso_px, reference_height_cm, height_px)
    
    if 'shoulder_width_cm' in measurements:
        chest_width = measurements['shoulder_width_cm']
        chest_depth = estimate_depth_from_width(chest_width, "chest")
        measurements['chest_circumference_cm'] = calculate_circumference_ellipse(chest_width, chest_depth)
    
    if 'shoulder_width_cm' in measurements:
        waist_width = measurements['shoulder_width_cm'] * 0.85
        waist_depth = estimate_depth_from_width(waist_width, "waist")
        measurements['waist_circumference_cm'] = calculate_circumference_ellipse(waist_width, waist_depth)
    
    if 'hip_width_cm' in measurements:
        hip_width = measurements['hip_width_cm']
        hip_depth = estimate_depth_from_width(hip_width, "hips")
        measurements['hip_circumference_cm'] = calculate_circumference_ellipse(hip_width, hip_depth)
    
    return measurements

# ==================== INTERFACE ====================

# Header
st.markdown("""
    <div class="header">
        <h1>👔 Tailleur Intelligent</h1>
        <p>Mesures Automatiques basées sur l'IA (YOLOv8)</p>
    </div>
""", unsafe_allow_html=True)

# Section Input
st.markdown("""
    <div class="input-section">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📏 Votre Taille**")
    height = st.number_input(
        "Taille en cm",
        min_value=140,
        max_value=220,
        value=170,
        step=1,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**👔 Type de Vêtement**")
    vêtement_choice = st.radio(
        "Choisir",
        ["👔 Veste", "🎨 Tissu Nigérian"],
        label_visibility="collapsed",
        horizontal=True
    )

with col3:
    st.markdown("**🌍 Tissu**")
    if "👔 Veste" in vêtement_choice:
        fabric_type = st.selectbox(
            "Tissu pour veste",
            ["Coton", "Polyester", "Laine", "Soie", "Lin"],
            label_visibility="collapsed"
        )
        garment_type = "Veste"
    else:
        fabric_type = st.selectbox(
            "Tissu nigérian",
            ["Ankara/Wax", "Aso Oke", "Bazin", "Adire", "Kente", "Lace"],
            label_visibility="collapsed"
        )
        garment_type = "Tissu Nigérian"

st.markdown("</div>", unsafe_allow_html=True)

# Upload section
st.markdown("""
    <div class="section-title">
        📤 Uploader votre Photo
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="info-box">
    <strong>✨ Recommandations:</strong><br>
    ✅ Debout, face caméra | ✅ Vue complète (tête aux pieds) | ✅ Bras écartés | ✅ Bonne lumière
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Traitement
if uploaded_file is not None:
    
    # Initialiser les variables
    garment_type = garment_type if 'garment_type' in locals() else "Veste"
    fabric_type = fabric_type if 'fabric_type' in locals() else "Coton"
    
    # Charger image
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Charger modèle
    model = load_yolo_model()
    
    # Détecter
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🤖 Détection de la pose...")
    progress_bar.progress(33)
    bbox, keypoints, annotated = detect_person_and_pose(image_cv, model)
    
    if bbox is None:
        st.error("❌ Aucune personne détectée. Vérifiez l'image.")
    else:
        status_text.text("📏 Calcul des mesures...")
        progress_bar.progress(66)
        measurements = calculate_body_measurements(image_cv, keypoints, height)
        
        if measurements is None:
            st.error("❌ Impossible de calculer les mesures.")
        else:
            progress_bar.progress(100)
            status_text.text("✅ Traitement terminé!")
            progress_bar.empty()
            status_text.empty()
            
            # Affichage résultats
            st.markdown("""
                <div class="section-title">
                    📊 Résultats
                </div>
            """, unsafe_allow_html=True)
            
            # Images
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.markdown("**📸 Image Originale**")
                st.image(image, use_column_width=True)
            
            with col_img2:
                st.markdown("**🦴 Pose Détectée**")
                st.image(annotated, channels="BGR", use_column_width=True)
            
            # Mesures
            st.markdown("""
                <div class="section-title">
                    📋 Mesures Corporelles
                </div>
            """, unsafe_allow_html=True)
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("""
                    <div class="measurement-card">
                        <div class="measurement-row">
                            <span class="measurement-label">📏 Hauteur totale</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">↔️ Largeur épaules</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">📐 Longueur torse</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">↔️ Largeur hanches</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                    </div>
                """.format(
                    measurements.get('height_total_cm', 0),
                    measurements.get('shoulder_width_cm', 0),
                    measurements.get('torso_length_cm', 0),
                    measurements.get('hip_width_cm', 0)
                ), unsafe_allow_html=True)
            
            with col_m2:
                st.markdown("""
                    <div class="measurement-card">
                        <div class="measurement-row">
                            <span class="measurement-label">💪 Bras gauche</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">💪 Bras droit</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">🦵 Jambe gauche</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                        <div class="measurement-row">
                            <span class="measurement-label">🦵 Jambe droite</span>
                            <span class="measurement-value">{:.1f} cm</span>
                        </div>
                    </div>
                """.format(
                    measurements.get('arm_length_left_cm', 0),
                    measurements.get('arm_length_right_cm', 0),
                    measurements.get('leg_length_left_cm', 0),
                    measurements.get('leg_length_right_cm', 0)
                ), unsafe_allow_html=True)
            
            # Tours de corps
            st.markdown("""
                <div class="section-title">
                    ⭕ Tours de Corps
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="measurement-card">
                    <div class="measurement-row">
                        <span class="measurement-label">⭕ Tour de poitrine</span>
                        <span class="measurement-value">{:.1f} cm</span>
                    </div>
                    <div class="measurement-row">
                        <span class="measurement-label">⭕ Tour de taille</span>
                        <span class="measurement-value">{:.1f} cm</span>
                    </div>
                    <div class="measurement-row">
                        <span class="measurement-label">⭕ Tour de hanches</span>
                        <span class="measurement-value">{:.1f} cm</span>
                    </div>
                </div>
            """.format(
                measurements.get('chest_circumference_cm', 0),
                measurements.get('waist_circumference_cm', 0),
                measurements.get('hip_circumference_cm', 0)
            ), unsafe_allow_html=True)
            
            # Recommandations
            st.markdown("""
                <div class="section-title">
                    🎯 Recommandations
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="recommendation-box">
                    <h3 style="margin-top: 0;">✨ Recommandations pour votre Tailleur</h3>
                    <p><strong>Type:</strong> {garment_type}</p>
                    <p><strong>Tissu:</strong> {fabric_type}</p>
                    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 0;">
                    <p><strong>Mesures clés à communiquer:</strong></p>
                    <ul>
                        <li>Tour poitrine: <strong>{measurements.get('chest_circumference_cm', 0):.1f} cm</strong></li>
                        <li>Tour taille: <strong>{measurements.get('waist_circumference_cm', 0):.1f} cm</strong></li>
                        <li>Tour hanches: <strong>{measurements.get('hip_circumference_cm', 0):.1f} cm</strong></li>
                        <li>Longueur bras: <strong>{(measurements.get('arm_length_left_cm', 0) + measurements.get('arm_length_right_cm', 0))/2:.1f} cm</strong></li>
                    </ul>
                    <p style="margin-top: 1rem; font-style: italic;">💡 Apportez ces mesures à votre tailleur pour un meilleur rendu!</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="success-box">
                    <strong>✅ Traitement réussi!</strong><br>
                    Mesures basées sur YOLOv8 • Marge d'erreur: ±4-6 cm
                </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="info-box">
            <strong>👆 Commencez par uploader une photo!</strong><br>
            Votre système de mesures intelligentes est prêt.
        </div>
    """, unsafe_allow_html=True)