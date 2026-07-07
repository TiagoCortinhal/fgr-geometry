from pathlib import Path

DATA = Path("/Users/tiago/PythonProject/fetal_growth_mechanism/data")
SEED = 0

# Join keys: visits_long* use fetus_id, impact_* use Cod. Both integer-aligned.
KEY_LONG = "fetus_id"
KEY_IMPACT = "Cod"

# Longitudinal biometry, 4 visits per fetus. INTERGROWTH-21 z-scores (visits_long_z.csv).
BIOM_Z = ["ac_z_ig21", "hc_z_ig21", "bpd_z_ig21", "fl_z_ig21", "efw_z_ig21"]
# Shape ratios (visits_long*).
RATIOS = ["hc_ac", "fl_ac"]

# Visit order, ga increasing. 'eco' is the late Doppler/cardiac snapshot visit.
VISITS = ["20s", "28s", "32s", "eco"]

# Late Doppler as per-fetus PERCENTILES (impact_features.csv), single late reading.
DOPPLER_PCTL = [
    "Percentil_CPR", "Percentil_AU", "Percentil_UTA",
    "Percentil_DV", "Percentil_ACM", "Percentil_Aortic_Ithsmus",
]
# Late cardiac percentiles (impact_features.csv), same late snapshot.
CARDIAC_PCTL = [
    "Percentil_Tapse", "Percentil_Mapse", "Percentil_Sapse", "Percentil_MPI",
    "Percentil_cardiac_area", "Percentil_LV_longitudinal", "Percentil_RV_longitudinal",
    "Percentil_LV_basal", "Percentil_RV_basal", "Percentil_septum",
    "Percentil_ICTms", "Percentil_ETms", "Percentil_IRTms",
]

# Visit blood pressure (impact_features.csv), per-fetus late-pregnancy readings.
BP = ["Vis2TR_diastolicBP", "Vis3TR_sistolicBP", "Vis3TR_diastolicBP"]

# Raw longitudinal Doppler PI/CPR (visits_long.csv). SPARSE at 28s/32s
# (~3-20% observed); the dense late reading is already in DOPPLER_PCTL. Optional.
RAW_DOPPLER = ["ua_pi", "mca_pi", "cpr", "uta_pi", "dv_pi"]
RAW_DOPPLER_VISITS = ["28s", "32s"]

# Per-fetus maternal covariates (constant across visits in visits_long).
MATERNAL = ["maternal_age", "maternal_height_cm", "maternal_weight_kg",
            "maternal_bmi", "nulliparous"]
# Maternal disease flags from impact_features.csv.
MATERNAL_DISEASE = ["HTAcronic", "DMpreg", "GDM"]

# Outcomes. Birth centile + binaries from visits_long; PE/preterm/NICU from impact_outcomes.
OUTCOMES = ["percentile_birth_pop", "sga", "severe_sga", "lga",
            "PEwithSGA", "PartoPret", "NICU"]


def ga_window(visit):
    """Nominal GA window (weeks) for each scan visit. eco = variable late snapshot."""
    return {
        "20s": (19, 24),
        "28s": (26, 30),
        "32s": (31, 34),
        "eco": (26, 39),
    }[visit]
