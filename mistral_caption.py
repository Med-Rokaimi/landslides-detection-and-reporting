#


import pandas as pd
import numpy as np
import os
import re
from itertools import combinations
import torch
from shapely.geometry import box
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
IOU_OVERLAP = 0.1
IOU_ADJACENT = 0.01
ADJACENCY_THRESHOLD = IOU_ADJACENT


# size categrisation 
SIZE_BUCKETS = {
    "very large": (3001, float('inf')),
    "large": (2001, 3000),
    "medium": (501, 2000),
    "small": (101, 500),
    "very small": (0, 100)
}

# -------------------------------------------------------------------
# Precomputed metadata
# -------------------------------------------------------------------
def compute_area_norm(bbox):
    return bbox[2] * bbox[3]

def size_bucket(area_norm):
    scaled = area_norm * IMAGE_AREA
    for name, (low, high) in SIZE_BUCKETS.items():
        if low < scaled <= high:
            return name
    return "unknown"

def make_shapely_box(bbox):
    x, y, w, h = bbox
    x0, y0 = x - w/2, y - h/2
    x1, y1 = x + w/2, y + h/2
    return box(x0, y0, x1, y1)

# -------------------------------------------------------------------
# Create summary per image
# -------------------------------------------------------------------
from itertools import combinations

def compute_summary(group, iou_thresh=IOU_OVERLAP, adjacency_thresh=ADJACENCY_THRESHOLD,
                    use_nonzero_intersection=True):
    """
    Returns pd.Series with fields:
      - intersections: list of overlap clusters (each cluster is a list of group.index values)
      - proximity: list of (idx_a, idx_b) pairs for very-close slides
      - num_slides: number of slides in the group
      - size_desc: string like "L:3, M:2"
    """
    idxs = list(group.index)
    boxes = [make_shapely_box((r.x_center, r.y_center, r.width, r.height))
             for r in group.itertuples()]

    n = len(boxes)
    counts = group['size'].value_counts().to_dict()
    size_desc = ", ".join(f"{k}:{v}" for k, v in counts.items())

    if n <= 1:
        return pd.Series({
            'intersections': [],
            'proximity': [],
            'num_slides': n,
            'size_desc': size_desc
        })

    # IoU helper
    def iou(b1, b2):
        inter = b1.intersection(b2).area
        union = b1.area + b2.area - inter
        return (inter / union) if (union > 0 and inter > 0) else 0.0

    # union-find for transitive clusters
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    proximity = []
    for i, j in combinations(range(n), 2):
        b1, b2 = boxes[i], boxes[j]
        inter_area = b1.intersection(b2).area
        iou_val = iou(b1, b2) if inter_area > 0 else 0.0

        # decide overlap (IoU or any nonzero intersection if enabled)
        if (iou_val >= iou_thresh) or (use_nonzero_intersection and inter_area > 0 and iou_val > 0):
            union(i, j)
        else:
            # not overlapping enough -> proximity check
            if b1.distance(b2) < adjacency_thresh:
                proximity.append((idxs[i], idxs[j]))

    # build clusters, keep only clusters with >1 member
    clusters = {}
    for pos, orig_idx in enumerate(idxs):
        root = find(pos)
        clusters.setdefault(root, []).append(orig_idx)
    intersections = [sorted(c) for c in clusters.values() if len(c) > 1]

    return pd.Series({
        'intersections': intersections,
        'proximity': proximity,
        'num_slides': n,
        'size_desc': size_desc
    })


# -------------------------------------------------------------------
# CONCLUSION 2
# -------------------------------------------------------------------
def precompute_conclusion(dem, ndvi, slope, lithology):
    """
    Precompute the conclusion based on the logic rules
    Returns the exact conclusion text that should be used.
    """
    # Convert to lowercase for case-insensitive comparison
    ndvi = str(ndvi).lower().strip()
    slope = str(slope).lower().strip()
    lith = str(lithology).lower().strip()
    dem = str(dem).lower().strip()
    
    
    if 'dense' in ndvi and 'moderate' in slope:
        return "Dense vegetation and moderate slope likely stabilized the terrain."
    elif ('low' in ndvi or 'no' in ndvi) and 'steep' in slope:
        return "Sparse vegetation with steep slope likely accelerated the landslide."
    elif 'high elevation' in dem and 'steep' in slope:
        return "High elevation with steep slopes increased landslide susceptibility."
    elif 'clay' in lith and 'vegetation' in ndvi:
        return "Clay-rich terrain and vegetation may have resisted landslide spread."
    elif 'sands' in lith and 'slope' in slope:
        return "Sandy slope terrain likely promoted flow-like landslides."
    elif 'pelagic' in lith and 'slope' in slope:
        return "Limestone terrain combined with slope triggered rockfalls or slides."
    elif 'alluvium' in lith and 'low' in ndvi:
        return "Alluvial deposits with low vegetation increased shallow landslide risk."
    else:
        return "Mixed terrain contributed variably to landslide behavior."
# -------------------------------------------------------------------
# Programmatic Section 2 Builder
# -------------------------------------------------------------------
def build_section2(group):
    lines = []
    for i, r in enumerate(group.itertuples(), 1):
        conclusion = precompute_conclusion(r.DEM, r.NDVI, r.SLOPE, r.LITH)
        lines.append(
            f"Slide No {i} is located in {r.zone} in an area of {r.DEM}, {r.SLOPE}, "
            f"composed of {r.LITH}, covered by {r.NDVI}. Conclusion: {conclusion}"
        )
    return "\n".join(lines)

# -------------------------------------------------------------------
# Summary Instruction
# -------------------------------------------------------------------
def get_summary_instruction(summary):
    summary_instruction = (
        "In one sentence, provide a concise synthesis reporting the number of slides and their general size categories (e.g., small/medium/large)." 
    )

    proximity = summary.get('proximity', [])
    intersections = summary.get('intersections', [])

    if summary.get('num_slides', 0) > 1:
        if intersections:
            summary_instruction += " If one or more slides exhibit overlapping footprints, state that overlapping footprints are present (do NOT provide exact counts of overlapping slides)."
        elif proximity:
            summary_instruction += " If slides are spatially adjacent or very close, mention proximity/adjacency in general terms (do NOT mention slide IDs or precise distances)."

    summary_instruction += "Do NOT include DEM, slope, lithology, NDVI, or zone into the summary. Don't add commentary or details."
    return summary_instruction
    
# -------------------------------------------------------------------
# Build Prompt
# -------------------------------------------------------------------
def build_prompt(image_id, summary, group):

    slide_conclusions = {}
    for i, r in enumerate(group.itertuples(), 1):
        conclusion = precompute_conclusion(r.DEM, r.NDVI, r.SLOPE, r.LITH)
        slide_conclusions[i] = conclusion
    
    # --------------------- PRIORITY RULES ---------------------
    priority_rules = r"""
    --- BEGIN PRIORITY RULES ---
        1) AUTHORITATIVE FIELDS: For determining whether slides "overlap" or are "adjacent", use ONLY the provided fields:
        - intersections  (list of index clusters)
        - proximity      (list of index pairs)
        Do NOT infer overlap/adjacency from DEM, slope, lithology, NDVI, size, coords, or any other attribute.

        2) DECISION LOGIC (follow exactly):
        - If intersections is a non-empty list -> set overlap_present = true
        - If intersections is empty and proximity contains pairs -> set overlap_present = false; adjacent_present = true
        - If both intersections and proximity are empty -> omit both keys ("overlap_present" and "adjacent_present") from the JSON
        - If any required input is missing or ambiguous -> set overlap_present = "unknown" (and explain briefly in the JSON "explanation" field)

        --- END PRIORITY RULES ---
"""
    summary_instruction = get_summary_instruction(summary)
    
    prompt = priority_rules + "\n\n"
    prompt += f"""You are a geo-spatial analyst and terrain expert. Generate a four-section caption for the following image using the provided attributes. Do NOT generate Section 2 (Geo-Spatial Features). Section 2 will be inserted programmatically.Focus only on Sections 1, 3, and 4."

    ImageID: {image_id}

    Summary information:
    - Number of slides: {summary['num_slides']}
    - Size distribution: {summary['size_desc']}
    - Proximity: {summary['proximity']}
    - Intersections: {summary['intersections']}
    

    Slide details with precomputed conclusions:
    """
    for i, r in enumerate(group.itertuples(), 1):
        prompt += (f"Slide{i}: coords={r.x_center:.4f} {r.y_center:.4f} {r.width:.4f} {r.height:.4f}; "
                f"size={r.size}; zone={r.zone}; DEM={r.DEM}; Slope={r.SLOPE}; Lithology={r.LITH}; NDVI={r.NDVI}; \n"
                f"Precomputed_Conclusion={slide_conclusions[i]}\n")


    prompt += f"""
    Follow this exact structure:

    1️⃣ Summary:
    {summary_instruction}
    
    2️⃣ Geo-Spatial Features:


    3️⃣ Observations and Analysis:
        Provide an overall, scene-level geologic interpretation that synthesizes **all slides together at once** (no per-slide enumeration). Keep it concise and grounded in DEM, slope, lithology, NDVI, and proximity/overlap.
        - Avoid repeating Section 2; focus on synthesis and expert judgment.
        
    4️⃣ Risk assessment:
        Write a **single paragraph** (no bullet points) that includes: (a) the overall risk Level — Low, Medium, or High — with a one-sentence justification **synthesized across all slides** based on DEM, slope, lithology, NDVI, and proximity;
        (b) the Probable velocity — choose one of extremely slow, very slow, slow, moderate, rapid, very rapid, or extremely rapid — with a brief rationale;
        and (c) the Conditional outlook for three scenarios expressed in prose (short, intense rainfall; prolonged wet season or snowmelt; vegetation loss such as NDVI decrease, harvest, or wildfire), each labeled only with Low/Medium/High likelihood within the paragraph (do **not** use scenario name headings or colons)

        Following some examples for section 3 Analysis:
        - Steep slopes at extremely high elevation within a weakly cemented clastic sequence (sands, sandstones, conglomerates) favor shallow translational failures with debris-slide transitions. Dense vegetation can obscure scarps and retain moisture, sustaining pore pressures. The overlapping footprints and their spread from upper-right through central-left to lower-left and mid-right indicate kinematic linkage along bedding, fracture, or contact-controlled weaknesses, with retrogressive upslope growth and lateral coalescence likely across the shared stratigraphic unit.
        - Carbonate-dominated terrain (pelagic limestones, marls, travertines) with steep to moderate slopes and overlapping slide footprints indicates failure along pre-existing discontinuities—bedding, joints, and marl partings—with local karstification enhancing detachment and channelizing runoff. Areas of low vegetation reduce root reinforcement and interception, while dense cover elsewhere can hide scarps and retain moisture. The spatial clustering and overlap point to kinematic linkage and sequential reactivation along shared slope breaks and daylighting beds.
        - Evaporitic lithologies (gypsum, anhydrite, salts) provide weak, dissolution-prone substrates where discontinuities and layering can daylight on inclined faces. Steep sectors supply the driving stresses for detachment, while dense vegetation adds root reinforcement and surface cohesion that locally counteracts movement, especially where slopes are only moderate. The overlapping arrangement implies kinematic linkage and short runout pathways, with reactivation or coalescence possible along fracture, bedding, or dissolution-enhanced planes within the same evaporitic package.
        - Steep slopes at high elevation within clay–marl units provide a cohesive but shear‐weak substrate that becomes particularly unstable when wet, promoting translational earth slides and local flowlike behavior. Dense vegetation across most sectors can mask scarps and add limited reinforcement on clay-rich ground, whereas the low-vegetation zone coincides with steeper faces that are more prone to movement. The clustering and overlap—especially in the right and lower sectors—suggest shared slope breaks and drainage pathways, with evidence of retrogression and coalescence as material from upslope sources connects to downslope accumulations.
        - Moderate slopes on moderately low terrain within a sandy–conglomeratic sequence favor shallow translational behavior only when moisture concentrates along contacts or coarse–fine transitions. Dense vegetation provides root reinforcement and interception that limits detachment and masks small scarps, and the clear separation of the two small features indicates localized, independent controls rather than a shared kinematic system or propagating failure.
        - Moderate slopes on moderately low terrain over evaporitic lithologies (gypsum, anhydrite, salts) point to weakness along dissolution-enhanced planes and discontinuities, with potential for subsidence and shallow translational movement where moisture concentrates. Dense vegetation provides root reinforcement and interception that generally restrain motion and may conceal minor scarps. The overlapping, clustered footprints suggest a shared hydrologic control—such as slope-parallel seepage or groundwater pathways—capable of coordinating reactivation and lateral coalescence within the same evaporitic package.

        Following some examples for section 4 Risk Assessment:
        - Overall risk is High, justified across all slides by the combination of steep relief at extremely high elevation, weakly cemented clastic lithologies, and overlapping footprints indicating kinematic linkage. The probable velocity is Rapid, as shallow failures in granular, weakly cemented materials on steep slopes tend to mobilize quickly once triggered. Likelihood is High during short, intense rainfall due to rapid pore-pressure rise in permeable clastics; High through prolonged wet periods or snowmelt as sustained saturation weakens contacts; and High following vegetation loss, where reduced root reinforcement on steep clastic slopes increases susceptibility and potential runout.
        - Overall risk is High, justified across all slides by steep to moderate slopes in jointed carbonate units, overlapping footprints that suggest connected kinematics, and sectors with low vegetation. The probable velocity is Rapid, reflecting blocky rockfall and translational slide behavior along fractures and weak marl interbeds on inclined faces, with susceptibility persisting on moderate slopes where bedding daylights and cover is sparse. Likelihood is High during short, intense rainfall due to rapid infiltration and joint lubrication; High through prolonged wet periods or snowmelt as sustained saturation weakens interfaces; and High following vegetation loss, where diminished root reinforcement and canopy interception further reduce stability and increase potential runout.
        - Overall risk is Medium, justified across all slides by the combination of weak evaporitic materials and overlapping footprints tempered by dense vegetation and a sector with only moderate slopes. The probable velocity is Moderate, reflecting shallow translational movements and block detachments controlled by discontinuities in soluble rocks, with vegetation limiting acceleration in less steep areas. Likelihood is High during short, intense rainfall due to rapid infiltration and dissolution-driven weakening; High during prolonged wet periods or snowmelt as sustained moisture degrades interfaces and raises pore pressures; and High following vegetation loss, when reduced root reinforcement and canopy interception diminish stability across the interconnected slide zones.
        - Overall risk is High, supported across all slides by the convergence of steep gradients at high elevation, clay–marl lithology, and overlapping footprints that indicate connected kinematics with possible retrogressive growth. The probable velocity is Moderate, reflecting translational movements in cohesive materials that can mobilize efficiently on steep faces while remaining partly constrained by matrix cohesion, with localized acceleration where saturation peaks. Likelihood is High during short, intense rainfall due to rapid pore-pressure rise and softening; High throughout prolonged wet periods or snowmelt as sustained moisture degrades shear strength; and High following vegetation loss, where reduced root reinforcement and interception further lower stability across the interconnected slide zones.
        - Overall risk is Low, justified across both slides by moderate gradients, dense vegetative cover, small feature size, and the absence of overlap that would indicate kinematic linkage, despite the inherently weaker, clastic lithologies. The probable velocity is Slow, reflecting shallow translational movements in weakly cemented sands and conglomerates on moderate slopes, tempered by root reinforcement. During short, intense rainfall the likelihood is Medium due to transient pore-pressure increases; during prolonged wet periods or snowmelt it is Medium as sustained moisture reduces shear strength; and following vegetation loss it is Medium because diminished root cohesion and canopy interception would lower stability even for these separated, small slides.
        - Overall risk is Medium, justified across all slides by stabilizing moderate slopes and dense cover counterbalanced by mechanically weak, dissolution-prone evaporites and overlapping footprints that imply connected kinematics. The probable velocity is Slow, consistent with shallow translational movements and subsidence governed by dissolution and bedding/joint daylighting, with localized speed-ups where wetting focuses. During short, intense rainfall the likelihood is High due to rapid softening and seepage along evaporitic contacts; through prolonged wet periods or snowmelt it is High as sustained moisture degrades strength and enlarges voids; after vegetation loss it is Medium because reduced root cohesion and interception would lower stability, though moderate gradients still temper runout.

        Do not add any extra sections or commentary."""
    return prompt

# -------------------------------------------------------------------
# Caption Generation
# -------------------------------------------------------------------
def generate_caption(prompt, model, tokenizer, device):
    full = f"""### Instruction:\n{prompt}\n### Response:"""
    inputs = tokenizer(full, return_tensors='pt').to(device)
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.split('### Response:')[-1].strip()

# -------------------------------------------------------------------
# Extract a section from generated text
# -------------------------------------------------------------------
def extract_section(text, marker):
    lines = text.splitlines()
    capture = []
    recording = False
    for line in lines:
        if line.strip().startswith(marker):
            recording = True
            continue
        if recording and line.strip().startswith(("1️⃣", "2️⃣", "3️⃣", "4️⃣")):
            break
        if recording:
            capture.append(line)
    return "\n".join(capture).strip()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset ='mistral_inputs.csv'
    output_dir = 'captions_output'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset)
    # Parse coords
    df[['x_center','y_center','width','height']] = df['coords'].str.split().apply(pd.Series).astype(float)
    df['area_norm'] = df.apply(lambda r: compute_area_norm((r.x_center, r.y_center, r.width, r.height)), axis=1)
    df['size'] = df['area_norm'].apply(size_bucket)

    # Compute summary per image
    summary = df.groupby('image').apply(compute_summary)
    df = df.merge(summary, on='image', how='left')

    # Load model
    model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    print("Model loaded successfully!")

    df['predicted_conclusion'] = ''
    df['precomputed_conclusion'] = ''

    # Loop and generate
    image_groups = list(df.groupby('image'))
    for img_id, grp in tqdm(image_groups, desc="Processing images", unit="image"):
        summ = summary.loc[img_id]
        
        # Precompute conclusions for this group
        for i, r in enumerate(grp.itertuples(), 1):
            precomputed_conclusion = precompute_conclusion(r.DEM, r.NDVI, r.SLOPE, r.LITH)
            df.at[r.Index, 'precomputed_conclusion'] = precomputed_conclusion
        
        prmpt = build_prompt(img_id, summ, grp)
        
        # Generate caption with progress indication
        tqdm.write(f"Generating caption for image: {img_id}")
        cap = generate_caption(prmpt, model, tokenizer, device)
        
        # Save caption
        #with open(f"{output_dir}/{img_id}_caption.txt", 'w') as f:
            #f.write(cap)
        
        section1 = extract_section(cap, "1️⃣")
        section3 = extract_section(cap, "3️⃣")
        section4 = extract_section(cap, "4️⃣")
        section2 = build_section2(grp)

        final_caption = (
            "1️⃣ Summary:\n" + section1 + "\n\n" +
            "2️⃣ Geo-Spatial Features:\n" + section2 + "\n\n" +
            "3️⃣ Observations and Analysis:\n" + section3 + "\n\n" +
            "4️⃣ Risk assessment:\n" + section4
        )

        with open(f"{output_dir}/{img_id}_caption.txt", 'w') as f:
            f.write(final_caption)

        
        # Save prompt
        #with open(f"{output_dir}/{img_id}_prompt.txt", 'w') as f:
            #f.write(prmpt)
        
        print(f" captiins are saved in {output_dir}")
        
        # Extract conclusions from generated text
        lines = re.findall(r"Conclusion:\s*(.*)", cap)
        for i, c in enumerate(lines):
            df.at[grp.index[i], 'predicted_conclusion'] = c
        
        
if __name__ == '__main__':
    main()