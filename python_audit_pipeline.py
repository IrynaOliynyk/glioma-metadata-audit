
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.gridspec import GridSpec
import pandas as pd

# The raw dataset was retrieved from WHO ICTRP on 03.01.2026.
# To replicate the study, download the data from Zenodo: https://doi.org/10.5281/zenodo.18430433
# Place the downloaded file 'glioma_who_3.01.2026.xlsx' in the same directory as this script.
import pandas as pd

file_path = r'glioma_who_3.01.2026.xlsx'

# Data Ingestion (N = 2357 records)
# keep_default_na=False is critical to preserve semantic markers like "N/A"
df = pd.read_excel(file_path, keep_default_na=False).head(2357)

# Columns selected for computational audit
columns_to_audit = ['Phase', 'Study_type']

for column_name in columns_to_audit:
    if column_name in df.columns:
        series = df[column_name]

        # Identify non-empty entries (excluding NaN and whitespace-only strings)
        non_empty_series = series[~series.isna() & (series.astype(str).str.strip() != '')]

        # Calculate frequency of unique descriptors
        value_counts = non_empty_series.value_counts()

        # Total count of detected text entries
        total_non_empty = value_counts.sum()

        # Calculation of Technical Voids (missing metadata)
        empty_cells_count = 2357 - total_non_empty

        # Terminal Output Formatting
        print(f"\nCOLUMN ANALYSIS: '{column_name}'")
        print("=" * 75)
        print(f"{'Value (Descriptor)':<60} | {'Count':>7}")
        print("-" * 75)

        for val, count in value_counts.items():
            # Clean string for display
            display_val = str(val).replace('\n', ' ').strip()
            if len(display_val) > 57:
                display_val = display_val[:57] + '..'
            print(f"{display_val:<60} | {count:>7}")

        print("-" * 75)
        print(f"TECHNICAL VOIDS (2357 - {total_non_empty}): {empty_cells_count}")
        print("=" * 75)

        # Create cleaned columns for downstream analysis
        # Entries not found in valid_values are explicitly labeled as 'empty'
        valid_values = set(value_counts.index)
        df[f'Cleaned_{column_name}'] = df[column_name].apply(
            lambda x: x if x in valid_values else 'empty'
        )

    else:
        print(f"Error: column '{column_name}' not found.")

# =========================================================
# 2. CLEAN PHASE
# =========================================================
# Phase Normalization Logic for Computational Audit
# =========================================================
def clean_phase(phase):
    # --- 1. Basic Normalization ---
    val = str(phase).strip()
    # Remove redundant whitespaces and convert to lowercase
    val_lower = " ".join(val.lower().split())

    if val == 'empty' or val_lower == '' or val_lower == 'nan':
        return 'Empty (Technical Void)'

    # --- 2. N/A Markers and Technical Noise ---
    na_markers = ['n/a', 'na', 'not applicable', 'not selected', 'none', 'not specified']
    if any(m == val_lower for m in na_markers) or 'not applicable' in val_lower:
        return 'Methodological Ambiguity (N/A)'

    # --- 3. Long Narrative Descriptions ---
    # Regex search for "Phase X" followed by a "yes/true" confirmation marker
    has_ph1 = re.search(r'phase\s*(i|1)\b[^y]*yes', val_lower)
    has_ph2 = re.search(r'phase\s*(ii|2)\b[^y]*yes', val_lower)
    has_ph3 = re.search(
        r'phase\s*(iii|3)\s*\)\s*:\s*(yes|true)\b',
        val_lower
    )
    has_ph4 = re.search(r'phase\s*(iv|4)\b[^y]*yes', val_lower)

    # First, check for hybrids within long descriptions (multiple "yes" markers)
    if (has_ph1 and has_ph2) or re.search(r'\b1\s*[\/\-\+]\s*2\b', val_lower):
        return 'Phase 1/2'
    if (has_ph2 and has_ph3) or re.search(r'\b2\s*[\/\-\+]\s*3\b', val_lower):
        return 'Phase 2/3'

    # Single "yes" markers in long descriptions
    if has_ph4: return 'Phase 4'
    if has_ph3: return 'Phase 3'
    if has_ph2: return 'Phase 2'
    if has_ph1: return 'Phase 1'

    # --- 4. Hybrids (Short entries and specific combinations) ---
    if re.search(
            r'\bphase\s*(1|i)\s*[,\/\-\+]\s*(phase\s*)?(2|ii)\b|\b(1|i)\s*[,\/\-\+]\s*(2|ii)\b',
            val_lower
    ):
        return 'Phase 1/2'

    if re.search(r'\bphase\s*(2|ii)\s*[,\/\-\+\s]+\s*(phase\s*)?(3|iii)\b|\b(2|ii)\s*[,\/\-\+]\s*(3|iii)\b', val_lower):
        return 'Phase 2/3'

    # --- 5. Strict Classical Phases ---

    # Phase 4 (Post-market and retrospective studies)
    if (
            (re.search(r'\b(4|iv)\b', val_lower) and ('phase' in val_lower or 'study' in val_lower)) or
            'post-market' in val_lower or
            val_lower == '4' or
            'postmarket' in val_lower or
            'retrospective study' in val_lower
    ):
        return 'Phase 4'

    # Phase 3
    if (
            val_lower == '3' or
            val_lower == 'iii' or
            re.search(r'\bphase\s*(3|iii)\b', val_lower) or
            re.search(r'\biii\s*\(phase\s*iii\s*study\)\b', val_lower) or  # matches "III (Phase III study)"
            (has_ph3 and not has_ph2)
    ):
        return 'Phase 3'

    # Phase 2
    if (
            val_lower == '2' or
            val_lower == 'ii' or
            re.search(r'\bphase\s*(2|ii)\b', val_lower) or
            'phase2' in val_lower or
            'phase-2' in val_lower
    ):
        return 'Phase 2'

    # Phase 1
    if (re.search(r'\b(1|i)\b', val_lower) and 'phase' in val_lower) or 'early phase 1' in val_lower:
        return 'Phase 1'

    # Phase 0 (Strict check)
    if val_lower == '0' or re.fullmatch(r'phase\s*0', val_lower):
        return 'Phase 0'

    # --- 6. Residual Narrative Entries ---
    if 'phase' in val_lower:
        return 'Over-specified Narrative'

    return 'Other / Structural Noise'

from collections import Counter, defaultdict

debug_bucket = defaultdict(list)

def clean_phase_debug(phase):
    result = clean_phase(phase)
    debug_bucket[result].append(str(phase))
    return result

df['Cleaned_Phase'] = df['Cleaned_Phase'].apply(clean_phase_debug)
phase_counts = Counter(df['Cleaned_Phase'])

print("\n=== PHASE COUNTS ===")
for k, v in phase_counts.items():
    print(f"{k:25s} : {v}")

print("\nTOTAL:", sum(phase_counts.values()))
print("\n=== PHASE 1/2 RAW VALUES ===")
for v in Counter(debug_bucket['Phase 1/2']).items():
    print(v)
print("\n=== PHASE 2 RAW VALUES ===")
for v in Counter(debug_bucket['Phase 2']).items():
    print(v)


# =========================================================
# 3. ASSIGN QUADRANTS
# =========================================================
def assign_quadrant(study_type):
    s = str(study_type).lower().strip()

    if any(k in s for k in ['interventional', 'medicinal product', 'diagnostic test', 'ba/be']):
        return 'Q1'
    if 'expanded access' in s:
        return 'Q4'
    if any(k in s for k in ['registry', 'database', 'record', 'retrospective']):
        return 'Q3'
    return 'Q2'
df['Quadrant'] = df['Study_type'].apply(assign_quadrant)





# =========================================================
# 4. COORDINATES
# =========================================================
coords = {'Q1': (1, 1), 'Q2': (-1, 1), 'Q3': (-1, -1), 'Q4': (1, -1)}
df['x'] = df['Quadrant'].apply(lambda q: coords[q][0] + np.random.uniform(-0.87, 0.87))
df['y'] = df['Quadrant'].apply(lambda q: coords[q][1] + np.random.uniform(-0.87, 0.87))

# =========================================================
# 5. COLORS
# =========================================================
color_map = {
    'Phase 0': '#B0B0B0',
    'Phase 1': '#1f77b4',
    'Phase 1/2': '#17becf',
    'Phase 2': '#2ca02c',
    'Phase 2/3': '#ff7f0e',
    'Phase 3': '#d62728',
    'Phase 4': '#8b0000',
    'Over-specified Narrative': '#7f7f7f',
    'Methodological Ambiguity (N/A)': '#ff1493',
    'Empty (Technical Void)': '#4b0082'
}

# =========================================================
# 6. LAYOUT
# =========================================================
fig = plt.figure(figsize=(22, 11))
gs = GridSpec(1, 3, width_ratios=[1.4, 4, 1.4])

ax_left = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[0, 1])
ax_right = fig.add_subplot(gs[0, 2])

for ax in [ax_left, ax_right]:
    ax.axis('off')

sns.set_style("white")

# =========================================================
# 7. MAIN SCATTER
# =========================================================
for phase, color in color_map.items():
    subset = df[df['Cleaned_Phase'] == phase]
    ax_main.scatter(
        subset['x'], subset['y'],
        c=color, label=phase,
        alpha=0.65, s=30,
        edgecolors='white', linewidths=0.2
    )

ax_main.axhline(0, color='black', linewidth=1.3)
ax_main.axvline(0, color='black', linewidth=1.3)
ax_main.set_xlim(-2, 2)
ax_main.set_ylim(-2, 2)
ax_main.set_xticks([])
ax_main.set_yticks([])

# =========================================================
# 8. QUADRANT INFO
# =========================================================
quad_info = {
    'Q1': 'INTERVENTIONAL',
    'Q2': 'OBSERVATIONAL',
    'Q3': 'PATIENT REGISTRIES',
    'Q4': 'EXPANDED ACCESS'
}

# =========================================================
# 9. SIDE STATISTICS
# =========================================================
def add_stats(ax, quadrants, align):
    y = 0.95
    for q in quadrants:
        q_df = df[df['Quadrant'] == q]

        ax.text(
            0.95 if align == 'right' else 0.05,
            y,
            f"{q} — {quad_info[q]}",
            fontsize=14, fontweight='bold',
            ha=align
        )
        y -= 0.06

        ax.text(
            0.95 if align == 'right' else 0.05,
            y,
            f"Total: {len(q_df)}",
            fontsize=12,
            ha=align,
            family='monospace'
        )
        y -= 0.06

        counts = q_df['Cleaned_Phase'].value_counts()
        for phase in color_map.keys():
            if phase in counts:
                ax.text(
                    0.95 if align == 'right' else 0.05,
                    y,
                    f"• {phase}: {counts[phase]}",
                    fontsize=10,
                    ha=align,
                    family='monospace'
                )
                y -= 0.040
        y -= 0.1


add_stats(ax_left, ['Q2', 'Q3'], align='right')
add_stats(ax_right, ['Q1', 'Q4'], align='left')

# =========================================================
# 10. LEGEND & TITLE
# =========================================================
handles, labels = ax_main.get_legend_handles_labels()

leg = fig.legend(
    handles, labels,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, 0.0),
    title='Reported Study Phases',
    title_fontsize=12,
    fontsize=11,
    frameon=True,
    labelcolor='black',
    borderpad=1,
    facecolor='white',
    edgecolor='black'
)


for handle in leg.legend_handles:
    handle.set_alpha(1.0)
    handle.set_sizes([120])

plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.05)
plt.suptitle(
    'Translational Phase-Space Mapping of Glioma Studies',
    fontsize=24, y=0.98
)

plt.show()



# --- UNIT 2: Visualization of structural phase distribution ---

def get_cleaned_pct_df(data, color_dict, threshold=0.1):
    total_counts = data['Cleaned_Phase'].value_counts(normalize=True) * 100
    valid_phases = total_counts[total_counts >= threshold].index.tolist()

    ordered_phases = [p for p in color_dict.keys() if p in valid_phases]

    all_quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    full_index = pd.MultiIndex.from_product([all_quadrants, ordered_phases],
                                            names=['Quadrant', 'Cleaned_Phase'])

    counts = data.groupby(['Quadrant', 'Cleaned_Phase']).size().reindex(full_index, fill_value=0).reset_index(
        name='count')
    q_totals = counts.groupby('Quadrant')['count'].transform('sum')
    counts['percentage'] = (counts['count'] / q_totals.replace(0, 1)) * 100

    return counts, ordered_phases


df_pct_filtered, final_phases = get_cleaned_pct_df(df, color_map)

# =========================================================
# 2. VISUALIZATION
# =========================================================
sns.set_theme(style="white")


fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=False)

# Map: Q2(0,0), Q1(0,1), Q3(1,0), Q4(1,1)
mapping = [
    (axes[0, 0], 'Q2'), (axes[0, 1], 'Q1'),
    (axes[1, 0], 'Q3'), (axes[1, 1], 'Q4')
]

for ax, q_name in mapping:
    subset = df_pct_filtered[df_pct_filtered['Quadrant'] == q_name]

    sns.barplot(
        data=subset, x="Cleaned_Phase", y="percentage",
        palette=color_map, order=final_phases, ax=ax,
        hue="Cleaned_Phase", legend=False
    )


    ax.set_title(f"Quadrant: {q_name}", fontsize=16, fontweight='bold', pad=15)


    max_val = subset['percentage'].max()
    ax.set_ylim(0, max_val * 1.16 if max_val > 0 else 100)

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel("")


    ax.set_xticks(range(len(final_phases)))
    ax.set_xticklabels(final_phases, rotation=45, ha='right', fontsize=10)


    if ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xticks(range(len(final_phases)))
        ax.set_xticklabels(final_phases, rotation=45, ha='right', fontsize=10)
    else:
        ax.set_xticklabels([])

# Annotations
    for p in ax.patches:
        h = p.get_height()
        if h > 0.01:
            ax.annotate(f'{h:.1f}%',
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='center',
                        xytext=(0, 9), textcoords='offset points',
                        fontsize=10, fontweight='bold')

plt.suptitle('Structural Distribution of Valid Study Phases (%)',
             fontsize=22, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.6, wspace=0.25)

plt.show()


# Calculation of the "Blindness Index" (Reporting Gap)
# ============================================================================
# CALCULATION OF THE REPORTING GAP (Completeness)
# ==

def calculate_phase_completeness (series):
    # Identify categories that we consider to be "blind spots" (missing data)

    gap_categories = [
        'Empty (Technical Void)',
        'Methodological Ambiguity (N/A)',
        'Other / Structural Noise',
        'Over-specified Narrative'
    ]

    # Count the number of useful entries (those not included in the list above)
    valid_entries = ~series.isin(gap_categories)
    return valid_entries.mean() * 100


# Group by quadrant and calculate the result
completeness_stats = df.groupby('Quadrant')['Cleaned_Phase'].apply(calculate_phase_completeness)

print("\n=== DATA COMPLETENESS (REPORTING GAP) ===")
for quad, val in completeness_stats.items():
    print(f"Quadrant {quad}: {val:.2f}% of entries have a defined Phase")

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    q_df = df[df['Quadrant'] == q]
    counts = q_df['Cleaned_Phase'].value_counts()

    # Maturity Calculation
    late_phases = counts.get('Phase 3', 0) + counts.get('Phase 4', 0)
    early_phases = counts.get('Phase 1', 0) + counts.get('Phase 1/2', 0) + counts.get('Phase 0', 0) + counts.get('Phase 2', 0) + counts.get('Phase 2/3', 0)
    maturity_ratio = late_phases / early_phases if early_phases > 0 else 0

    print(f"Quadrant {q} Maturity Ratio: {maturity_ratio:.2f}")

# =========================================================================
# ANALYSIS OF REAL RESEARCH OBJECTS (PHASE 4)
# ==

# 1. Filter only phase 4
ph4_data = df[df['Cleaned_Phase'] == 'Phase 4'].copy()

# 2. Select key columns to check
# Use Public_title as the main source of titles

analysis_cols = ['Quadrant', 'Public_title', 'Study_type']
results = ph4_data[analysis_cols]

print(f"Phase 4 records found: {len(results)}")
print("-" * 30)

# 3. Output the list for viewing
# We group by quadrant to see the difference between Q1 and Q2
for quad in ['Q1', 'Q2']:
    quad_subset = results[results['Quadrant'] == quad]
    print(f"\nQUADRANT {quad} (n={len(quad_subset)}):")

    if not quad_subset.empty:
        # Output unique titles to filter out duplicates
        unique_titles = quad_subset['Public_title'].unique()
        for i, title in enumerate(unique_titles[:100], 1):  # Let's watch all 21
            print(f"  {i}. {title[:1550]}...")
    else:

        print("  No records found.")
