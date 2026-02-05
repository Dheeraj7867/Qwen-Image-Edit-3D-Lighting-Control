import gradio as gr
import numpy as np
import random
import torch
import spaces
from typing import Iterable
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

MAX_SEED = np.iinfo(np.int32).max

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4",
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)
try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

ADAPTER_SPECS = {
    "Multi-Angle-Lighting": {
        "repo": "dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
        "weights": "多角度灯光-251116.safetensors",
        "adapter_name": "multi-angle-lighting"
    },
}
loaded = False

AZIMUTH_MAP = {
    0: "Front",
    45: "Right Front",
    90: "Right",
    135: "Right Rear",
    180: "Rear",
    225: "Left Rear",
    270: "Left",
    315: "Left Front"
}

ELEVATION_MAP = {
    -90: "Below",
    0: "",
    90: "Above"
}

def snap_to_nearest(value, options):
    """Snap a value to the nearest option in a list."""
    return min(options, key=lambda x: abs(x - value))

def build_lighting_prompt(azimuth: float, elevation: float) -> str:
    """
    Build a lighting prompt from azimuth and elevation values.
    """
    azimuth_snapped = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    
    if elevation_snapped == 0:
        return f"Light source from the {AZIMUTH_MAP[azimuth_snapped]}"
    else:
        return f"Light source from {ELEVATION_MAP[elevation_snapped]}"

@spaces.GPU
def infer_lighting_edit(
    image: Image.Image,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
):
    global loaded
    progress = gr.Progress(track_tqdm=True)
    
    if not loaded:
        pipe.load_lora_weights(
            ADAPTER_SPECS["Multi-Angle-Lighting"]["repo"],
            weight_name=ADAPTER_SPECS["Multi-Angle-Lighting"]["weights"],
            adapter_name=ADAPTER_SPECS["Multi-Angle-Lighting"]["adapter_name"]
        )
        pipe.set_adapters([ADAPTER_SPECS["Multi-Angle-Lighting"]["adapter_name"]], adapter_weights=[1.0])
        loaded = True
    
    prompt = build_lighting_prompt(azimuth, elevation)
    print(f"Generated Prompt: {prompt}")

    progress(0.7, desc="Fast lighting enabled....")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    if image is None:
        raise gr.Error("Please upload an image first.")
    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    result = pipe(
        image=[pil_image],
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]
    return result, seed, prompt

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    return new_width, new_height

class LightingControl3D(gr.HTML):
    """
    A 3D lighting control component using Three.js.
    """
    def __init__(self, value=None, imageUrl=None, **kwargs):
        if value is None:
            value = {"azimuth": 0, "elevation": 0}
        
        html_template = """
        <div id="lighting-control-wrapper" style="width: 100%; height: 450px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
            <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10;"></div>
        </div>
        """
        
        js_on_load = """
        (() => {
            const wrapper = element.querySelector('#lighting-control-wrapper');
            const promptOverlay = element.querySelector('#prompt-overlay');
            
            const initScene = () => {
                if (typeof THREE === 'undefined') {
                    setTimeout(initScene, 100);
                    return;
                }
                
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
                camera.position.set(4.5, 3, 4.5);
                camera.lookAt(0, 0.75, 0);
                
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                wrapper.insertBefore(renderer.domElement, promptOverlay);
                
                scene.add(new THREE.AmbientLight(0xffffff, 0.1));
                
                const ground = new THREE.Mesh(
                    new THREE.PlaneGeometry(10, 10),
                    new THREE.ShadowMaterial({ opacity: 0.3 })
                );
                ground.rotation.x = -Math.PI / 2;
                ground.position.y = 0;
                ground.receiveShadow = true;
                scene.add(ground);
                
                scene.add(new THREE.GridHelper(8, 16, 0x333333, 0x222222));
                
                const CENTER = new THREE.Vector3(0, 0.75, 0);
                const BASE_DISTANCE = 2.5;
                const AZIMUTH_RADIUS = 2.4;
                const ELEVATION_RADIUS = 1.8;
                
                let azimuthAngle = props.value?.azimuth || 0;
                let elevationAngle = props.value?.elevation || 0;
                
                const azimuthSteps = [0, 45, 90, 135, 180, 225, 270, 315];
                const elevationSteps = [-90, 0, 90];
                const azimuthNames = {
                    0: 'Front', 45: 'Right Front', 90: 'Right',
                    135: 'Right Rear', 180: 'Rear', 225: 'Left Rear',
                    270: 'Left', 315: 'Left Front'
                };
                const elevationNames = { '-90': 'Below', '0': '', '90': 'Above' };
                
                function snapToNearest(value, steps) {
                    return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
                }
                
                function createPlaceholderTexture() {
                    const canvas = document.createElement('canvas');
                    canvas.width = 256;
                    canvas.height = 256;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#3a3a4a';
                    ctx.fillRect(0, 0, 256, 256);
                    ctx.fillStyle = '#ffcc99';
                    ctx.beginPath();
                    ctx.arc(128, 128, 80, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#333';
                    ctx.beginPath();
                    ctx.arc(100, 110, 10, 0, Math.PI * 2);
                    ctx.arc(156, 110, 10, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
                    ctx.stroke();
                    return new THREE.CanvasTexture(canvas);
                }
                
                let currentTexture = createPlaceholderTexture();
                const planeMaterial = new THREE.MeshStandardMaterial({ map: currentTexture, side: THREE.DoubleSide, roughness: 0.5, metalness: 0 });
                let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                targetPlane.receiveShadow = true;
                scene.add(targetPlane);
                
                function updateTextureFromUrl(url) {
                    if (!url) {
                        planeMaterial.map = createPlaceholderTexture();
                        planeMaterial.needsUpdate = true;
                        scene.remove(targetPlane);
                        targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                        targetPlane.position.copy(CENTER);
                        targetPlane.receiveShadow = true;
                        scene.add(targetPlane);
                        return;
                    }
                    
                    const loader = new THREE.TextureLoader();
                    loader.crossOrigin = 'anonymous';
                    loader.load(url, (texture) => {
                        texture.minFilter = THREE.LinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        planeMaterial.map = texture;
                        planeMaterial.needsUpdate = true;
                        
                        const img = texture.image;
                        if (img && img.width && img.height) {
                            const aspect = img.width / img.height;
                            const maxSize = 1.5;
                            let planeWidth, planeHeight;
                            if (aspect > 1) {
                                planeWidth = maxSize;
                                planeHeight = maxSize / aspect;
                            } else {
                                planeHeight = maxSize;
                                planeWidth = maxSize * aspect;
                            }
                            scene.remove(targetPlane);
                            targetPlane = new THREE.Mesh(
                                new THREE.PlaneGeometry(planeWidth, planeHeight),
                                planeMaterial
                            );
                            targetPlane.position.copy(CENTER);
                            targetPlane.receiveShadow = true;
                            scene.add(targetPlane);
                        }
                    }, undefined, (err) => {
                        console.error('Failed to load texture:', err);
                    });
                }
                
                if (props.imageUrl) {
                    updateTextureFromUrl(props.imageUrl);
                }
                
                // --- NEW LIGHT MODEL: SQUARE STUDIO LIGHT WITH RAYS ---
                const lightGroup = new THREE.Group();

                // 1. Studio Panel Housing (Black, thin, square)
                const panelGeo = new THREE.BoxGeometry(0.8, 0.8, 0.1);
                const panelMat = new THREE.MeshStandardMaterial({ 
                    color: 0x111111,    // Black body
                    roughness: 0.3,
                    metalness: 0.8 
                });
                const panel = new THREE.Mesh(panelGeo, panelMat);
                // Shift box slightly back so the front face is at z=0 relative to the group
                panel.position.z = -0.05; 
                lightGroup.add(panel);
                
                // 2. Emissive Light Face (Bright White)
                const faceGeo = new THREE.PlaneGeometry(0.75, 0.75);
                const faceMat = new THREE.MeshBasicMaterial({ 
                    color: 0xffffff,     // Pure white
                    side: THREE.DoubleSide
                });
                const face = new THREE.Mesh(faceGeo, faceMat);
                face.position.z = 0.01; // Slightly in front of the black housing
                lightGroup.add(face);
                
                // 3. Volumetric Light Rays (Transparent Cone)
                // CylinderGeometry(radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded)
                const beamHeight = 4.0;
                const beamGeo = new THREE.CylinderGeometry(0.38, 1.2, beamHeight, 32, 1, true);
                
                // Rotate cylinder to point along +Z axis
                beamGeo.rotateX(-Math.PI / 2); 
                // Translate so the top (start) of the beam sits on the light face
                beamGeo.translate(0, 0, beamHeight / 2); 
                
                const beamMat = new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    transparent: true,
                    opacity: 0.12,          // Low opacity for subtleness
                    side: THREE.DoubleSide,
                    depthWrite: false,      // Important for transparent sorting
                    blending: THREE.AdditiveBlending // Glow effect
                });
                
                const beam = new THREE.Mesh(beamGeo, beamMat);
                lightGroup.add(beam);

                // Actual SpotLight Calculation Source
                const spotLight = new THREE.SpotLight(0xffffff, 10, 10, Math.PI / 3, 1, 1);
                spotLight.position.set(0, 0, 0); // Position at the center of the custom mesh
                spotLight.castShadow = true;
                spotLight.shadow.mapSize.width = 1024;
                spotLight.shadow.mapSize.height = 1024;
                spotLight.shadow.camera.near = 0.5;
                spotLight.shadow.camera.far = 500;
                spotLight.shadow.bias = -0.005;
                lightGroup.add(spotLight);
                
                const lightTarget = new THREE.Object3D();
                lightTarget.position.copy(CENTER);
                scene.add(lightTarget);
                spotLight.target = lightTarget;
                
                scene.add(lightGroup);
                
                // --- CONTROLS ---
                
                const azimuthRing = new THREE.Mesh(
                    new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 64),
                    new THREE.MeshStandardMaterial({ color: 0xffff00, emissive: 0xffff00, emissiveIntensity: 0.3 })
                );
                azimuthRing.rotation.x = Math.PI / 2;
                azimuthRing.position.y = 0.05;
                scene.add(azimuthRing);
                
                const azimuthHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xffff00, emissive: 0xffff00, emissiveIntensity: 0.5 })
                );
                azimuthHandle.userData.type = 'azimuth';
                scene.add(azimuthHandle);
                
                const arcPoints = [];
                for (let i = 0; i <= 32; i++) {
                    const angle = THREE.MathUtils.degToRad(-90 + (180 * i / 32));
                    arcPoints.push(new THREE.Vector3(-0.8, ELEVATION_RADIUS * Math.sin(angle) + CENTER.y, ELEVATION_RADIUS * Math.cos(angle)));
                }
                const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
                const elevationArc = new THREE.Mesh(
                    new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false),
                    new THREE.MeshStandardMaterial({ color: 0x0000ff, emissive: 0x0000ff, emissiveIntensity: 0.3 })
                );
                scene.add(elevationArc);
                
                const elevationHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0x0000ff, emissive: 0x0000ff, emissiveIntensity: 0.5 })
                );
                elevationHandle.userData.type = 'elevation';
                scene.add(elevationHandle);
                
                const refreshBtn = document.createElement('button');
                refreshBtn.innerHTML = 'Reset View';
                refreshBtn.style.position = 'absolute';
                refreshBtn.style.top = '15px';
                refreshBtn.style.right = '15px';
                refreshBtn.style.background = '#e63e00';
                refreshBtn.style.color = '#fff';
                refreshBtn.style.border = 'none';
                refreshBtn.style.padding = '8px 16px';
                refreshBtn.style.borderRadius = '6px';
                refreshBtn.style.cursor = 'pointer';
                refreshBtn.style.zIndex = '10';
                refreshBtn.style.fontSize = '14px';
                refreshBtn.style.fontWeight = '600';
                refreshBtn.style.fontFamily = 'system-ui, sans-serif';
                refreshBtn.style.boxShadow = '0 2px 5px rgba(0,0,0,0.3)';
                refreshBtn.style.transition = 'background 0.2s';
                
                refreshBtn.onmouseover = () => refreshBtn.style.background = '#ff5722';
                refreshBtn.onmouseout = () => refreshBtn.style.background = '#e63e00';
                
                wrapper.appendChild(refreshBtn);
                
                refreshBtn.addEventListener('click', () => {
                    azimuthAngle = 0;
                    elevationAngle = 0;
                    updatePositions();
                    updatePropsAndTrigger();
                });
                
                function updatePositions() {
                    const distance = BASE_DISTANCE;
                    const azRad = THREE.MathUtils.degToRad(azimuthAngle);
                    const elRad = THREE.MathUtils.degToRad(elevationAngle);
                    
                    const lightX = distance * Math.sin(azRad) * Math.cos(elRad);
                    const lightY = distance * Math.sin(elRad) + CENTER.y;
                    const lightZ = distance * Math.cos(azRad) * Math.cos(elRad);
                    
                    lightGroup.position.set(lightX, lightY, lightZ);
                    lightGroup.lookAt(CENTER);
                    
                    azimuthHandle.position.set(AZIMUTH_RADIUS * Math.sin(azRad), 0.05, AZIMUTH_RADIUS * Math.cos(azRad));
                    elevationHandle.position.set(-0.8, ELEVATION_RADIUS * Math.sin(elRad) + CENTER.y, ELEVATION_RADIUS * Math.cos(elRad));
                    
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    let prompt = 'Light source from';
                    if (elSnap !== 0) {
                        prompt += ' ' + elevationNames[String(elSnap)];
                    } else {
                        prompt += ' the ' + azimuthNames[azSnap];
                    }
                    promptOverlay.textContent = prompt;
                }
                
                function updatePropsAndTrigger() {
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    
                    props.value = { azimuth: azSnap, elevation: elSnap };
                    trigger('change', props.value);
                }
                
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                let isDragging = false;
                let dragTarget = null;
                let dragStartMouse = new THREE.Vector2();
                const intersection = new THREE.Vector3();
                
                const canvas = renderer.domElement;
                
                canvas.addEventListener('mousedown', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                        canvas.style.cursor = 'grabbing';
                    }
                });
                
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -90, 90);
                            }
                        }
                        updatePositions();
                    } else {
                        raycaster.setFromCamera(mouse, camera);
                        const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle]);
                        [azimuthHandle, elevationHandle].forEach(h => {
                            h.material.emissiveIntensity = 0.5;
                            h.scale.setScalar(1);
                        });
                        if (intersects.length > 0) {
                            intersects[0].object.material.emissiveIntensity = 0.8;
                            intersects[0].object.scale.setScalar(1.1);
                            canvas.style.cursor = 'grab';
                        } else {
                            canvas.style.cursor = 'default';
                        }
                    }
                });
                
                const onMouseUp = () => {
                    if (dragTarget) {
                        dragTarget.material.emissiveIntensity = 0.5;
                        dragTarget.scale.setScalar(1);
                        
                        const targetAz = snapToNearest(azimuthAngle, azimuthSteps);
                        const targetEl = snapToNearest(elevationAngle, elevationSteps);
                        
                        const startAz = azimuthAngle, startEl = elevationAngle;
                        const startTime = Date.now();
                        
                        function animateSnap() {
                            const t = Math.min((Date.now() - startTime) / 200, 1);
                            const ease = 1 - Math.pow(1 - t, 3);
                            
                            let azDiff = targetAz - startAz;
                            if (azDiff > 180) azDiff -= 360;
                            if (azDiff < -180) azDiff += 360;
                            azimuthAngle = startAz + azDiff * ease;
                            if (azimuthAngle < 0) azimuthAngle += 360;
                            if (azimuthAngle >= 360) azimuthAngle -= 360;
                            
                            elevationAngle = startEl + (targetEl - startEl) * ease;
                            
                            updatePositions();
                            if (t < 1) requestAnimationFrame(animateSnap);
                            else updatePropsAndTrigger();
                        }
                        animateSnap();
                    }
                    isDragging = false;
                    dragTarget = null;
                    canvas.style.cursor = 'default';
                };
                
                canvas.addEventListener('mouseup', onMouseUp);
                canvas.addEventListener('mouseleave', onMouseUp);
                canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -90, 90);
                            }
                        }
                        updatePositions();
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                canvas.addEventListener('touchcancel', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                updatePositions();
                
                function render() {
                    requestAnimationFrame(render);
                    renderer.render(scene, camera);
                }
                render();
                
                new ResizeObserver(() => {
                    camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                }).observe(wrapper);
                
                wrapper._updateFromProps = (newVal) => {
                    if (newVal && typeof newVal === 'object') {
                        azimuthAngle = newVal.azimuth ?? azimuthAngle;
                        elevationAngle = newVal.elevation ?? elevationAngle;
                        updatePositions();
                    }
                };
                
                wrapper._updateTexture = updateTextureFromUrl;
                
                let lastImageUrl = props.imageUrl;
                let lastValue = JSON.stringify(props.value);
                setInterval(() => {
                    if (props.imageUrl !== lastImageUrl) {
                        lastImageUrl = props.imageUrl;
                        updateTextureFromUrl(props.imageUrl);
                    }
                    const currentValue = JSON.stringify(props.value);
                    if (currentValue !== lastValue) {
                        lastValue = currentValue;
                        if (props.value && typeof props.value === 'object') {
                            azimuthAngle = props.value.azimuth ?? azimuthAngle;
                            elevationAngle = props.value.elevation ?? elevationAngle;
                            updatePositions();
                        }
                    }
                }, 100);
            };
            
            initScene();
        })();
        """
        
        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            imageUrl=imageUrl,
            **kwargs
        )

css="""
#col-container { max-width: 1200px; margin: 0 auto; }
.dark .progress-text { color: white !important; }
#lighting-3d-control { min-height: 450px; }
.slider-row { display: flex; gap: 10px; align-items: center; }
#main-title h1 {font-size: 2.4em !important;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **Qwen-Image-Edit-3D-Lighting-Control**", elem_id="main-title")
    gr.Markdown("Control lighting directions using the 3D viewport or sliders. Using the [Multi-Angle-Lighting](https://huggingface.co/dx8152/Qwen-Edit-2509-Multi-Angle-Lighting) LoRA for precise lighting control.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Input Image", type="pil", height=300)
            
            gr.Markdown("### 3D Lighting Control")

            lighting_3d = LightingControl3D(
                value={"azimuth": 0, "elevation": 0},
                elem_id="lighting-3d-control"
            )
            run_btn = gr.Button("Generate Image", variant="primary", size="lg")
            
            gr.Markdown("### Slider Controls")
            
            azimuth_slider = gr.Slider(
                label="Azimuth (Horizontal Rotation)",
                minimum=0,
                maximum=315,
                step=45,
                value=0,
                info="0°=front, 90°=right, 180°=rear, 270°=left"
            )
            
            elevation_slider = gr.Slider(
                label="Elevation (Vertical Angle)",
                minimum=-90,
                maximum=90,
                step=90,
                value=0,
                info="-90°=from below, 0°=horizontal, 90°=from above"
            )

            with gr.Row():
                prompt_preview = gr.Textbox(
                    label="Generated Prompt",
                    value="Light source from the Front",
                    interactive=True,
                    lines=1,
                )
        
        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", height=555)
            
            with gr.Accordion("Advanced Settings", open=True):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=20, step=1, value=4)
                height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)

            with gr.Accordion("About the Space", open=True):
                gr.Markdown(
                    "This app, *Qwen-Image-Edit-3D-Lighting-Control*, is designed by [prithivMLmods](https://huggingface.co/prithivMLmods) to accelerate fast inference with 4-step image edits and is inspired by [qwen-image-multiple-angles-3d-camera](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera). For more adapters, visit: [Qwen-Image-Edit-LoRAs](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image-Edit-2509)."
                )
            
    def update_prompt_from_sliders(azimuth, elevation):
        """Update prompt preview when sliders change."""
        prompt = build_lighting_prompt(azimuth, elevation)
        return prompt
    
    def sync_3d_to_sliders(lighting_value):
        """Sync 3D control changes to sliders."""
        if lighting_value and isinstance(lighting_value, dict):
            az = lighting_value.get('azimuth', 0)
            el = lighting_value.get('elevation', 0)
            prompt = build_lighting_prompt(az, el)
            return az, el, prompt
        return gr.update(), gr.update(), gr.update()
    
    def sync_sliders_to_3d(azimuth, elevation):
        """Sync slider changes to 3D control."""
        return {"azimuth": azimuth, "elevation": elevation}
    
    def update_3d_image(image):
        """Update the 3D component with the uploaded image."""
        if image is None:
            return gr.update(imageUrl=None)

        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return gr.update(imageUrl=data_url)
    
    for slider in [azimuth_slider, elevation_slider]:
        slider.change(
            fn=update_prompt_from_sliders,
            inputs=[azimuth_slider, elevation_slider],
            outputs=[prompt_preview]
        )
    
    lighting_3d.change(
        fn=sync_3d_to_sliders,
        inputs=[lighting_3d],
        outputs=[azimuth_slider, elevation_slider, prompt_preview]
    )
    
    for slider in [azimuth_slider, elevation_slider]:
        slider.release(
            fn=sync_sliders_to_3d,
            inputs=[azimuth_slider, elevation_slider],
            outputs=[lighting_3d]
        )
    
    run_btn.click(
        fn=infer_lighting_edit,
        inputs=[image, azimuth_slider, elevation_slider, seed, randomize_seed, guidance_scale, num_inference_steps, height, width],
        outputs=[result, seed, prompt_preview]
    )
    
    image.upload(
        fn=update_dimensions_on_upload,
        inputs=[image],
        outputs=[width, height]
    ).then(
        fn=update_3d_image,
        inputs=[image],
        outputs=[lighting_3d]
    )
    
    image.clear(
        fn=lambda: gr.update(imageUrl=None),
        outputs=[lighting_3d]
    )
    
if __name__ == "__main__":
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    css = '.fillable{max-width: 1200px !important}'
    demo.launch(head=head, css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)
